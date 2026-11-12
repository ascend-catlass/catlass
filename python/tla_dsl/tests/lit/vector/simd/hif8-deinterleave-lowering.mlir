// RUN: %tla_compile %s -o - 2>&1 | %filecheck %s

!fsrc = !tla.tensor<!tla.layout<!tla.shape<128>, !tla.stride<1>, !tla.shape<128>, RowMajor>, !tla.coord<0>, !tla.ptr<f32, ub, 4>>
!fvec = !tla.tensor<!tla.layout<!tla.shape<64>, !tla.stride<1>, !tla.shape<64>, RowMajor>, !tla.coord<0>, !tla.ptr<f32, ub, 4>>

module {
  func.func @vector_load_dintlv_b32(
      %src_memref: memref<128xf32, #hivm.address_space<ub>>,
      %dst0_memref: memref<64xf32, #hivm.address_space<ub>>,
      %dst1_memref: memref<64xf32, #hivm.address_space<ub>>) {
    %src_c0 = arith.constant 0 : index
    %src_c1 = arith.constant 1 : index
    %src_c128 = arith.constant 128 : index
    %src = tla.tensor_desc %src_memref shape [%src_c1, %src_c128, %src_c1, %src_c1] stride [%src_c128, %src_c1, %src_c1, %src_c1] origin_shape [%src_c1, %src_c128] coord [%src_c0, %src_c0] : memref<128xf32, #hivm.address_space<ub>> -> !fsrc
    %dst0_c0 = arith.constant 0 : index
    %dst0_c1 = arith.constant 1 : index
    %dst0_c64 = arith.constant 64 : index
    %dst0 = tla.tensor_desc %dst0_memref shape [%dst0_c1, %dst0_c64, %dst0_c1, %dst0_c1] stride [%dst0_c64, %dst0_c1, %dst0_c1, %dst0_c1] origin_shape [%dst0_c1, %dst0_c64] coord [%dst0_c0, %dst0_c0] : memref<64xf32, #hivm.address_space<ub>> -> !fvec
    %dst1_c0 = arith.constant 0 : index
    %dst1_c1 = arith.constant 1 : index
    %dst1_c64 = arith.constant 64 : index
    %dst1 = tla.tensor_desc %dst1_memref shape [%dst1_c1, %dst1_c64, %dst1_c1, %dst1_c1] stride [%dst1_c64, %dst1_c1, %dst1_c1, %dst1_c1] origin_shape [%dst1_c1, %dst1_c64] coord [%dst1_c0, %dst1_c0] : memref<64xf32, #hivm.address_space<ub>> -> !fvec
    "tla.vec.func"() ({
      %shape128 = "tla.make_shape"() : () -> !tla.shape<128>
      %shape64 = "tla.make_shape"() : () -> !tla.shape<64>
      %coord = "tla.make_coord"() : () -> !tla.coord<0>
      %src_tile = "tla.tile_view"(%src, %shape128, %coord) : (!fsrc, !tla.shape<128>, !tla.coord<0>) -> !fsrc
      %dst0_tile = "tla.tile_view"(%dst0, %shape64, %coord) : (!fvec, !tla.shape<64>, !tla.coord<0>) -> !fvec
      %dst1_tile = "tla.tile_view"(%dst1, %shape64, %coord) : (!fvec, !tla.shape<64>, !tla.coord<0>) -> !fvec
      %v0, %v1 = tla.load %src_tile {load_dist = #tla.load_dist<dintlv_b32>} : !fsrc -> !tla.vector<64xf32>, !tla.vector<64xf32>
      %encoded = "tla.cast"(%v0) <{trait = array<i32: 0, 1, 0>}> : (!tla.vector<64xf32>) -> !tla.vector<256x!tla.hif8>
      %decoded = "tla.cast"(%encoded) <{trait = array<i32: 0, 0, 0>}> : (!tla.vector<256x!tla.hif8>) -> !tla.vector<64xf32>
      tla.store %dst0_tile, %decoded : !fvec, !tla.vector<64xf32>
      tla.store %dst1_tile, %v1 : !fvec, !tla.vector<64xf32>
    }) : () -> ()
    return
  }

  // Descriptor coordinates materialize a subview at eight f32 elements (32 bytes).
  func.func @vector_load_dintlv_b32_offset(
      %src_memref: memref<?xf32, #hivm.address_space<ub>>,
      %dst_memref: memref<64xf32, #hivm.address_space<ub>>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c64 = arith.constant 64 : index
    %c128 = arith.constant 128 : index
    %c8 = arith.constant 8 : index
    %src = tla.tensor_desc %src_memref shape [%c1, %c128, %c1, %c1] stride [%c128, %c1, %c1, %c1] origin_shape [%c1, %c128] coord [%c0, %c8] : memref<?xf32, #hivm.address_space<ub>> -> !fsrc
    %dst = tla.tensor_desc %dst_memref shape [%c1, %c64, %c1, %c1] stride [%c64, %c1, %c1, %c1] origin_shape [%c1, %c64] coord [%c0, %c0] : memref<64xf32, #hivm.address_space<ub>> -> !fvec
    "tla.vec.func"() ({
      %shape128 = "tla.make_shape"() : () -> !tla.shape<128>
      %shape64 = "tla.make_shape"() : () -> !tla.shape<64>
      %coord = "tla.make_coord"() : () -> !tla.coord<0>
      %src_tile = "tla.tile_view"(%src, %shape128, %coord) : (!fsrc, !tla.shape<128>, !tla.coord<0>) -> !fsrc
      %dst_tile = "tla.tile_view"(%dst, %shape64, %coord) : (!fvec, !tla.shape<64>, !tla.coord<0>) -> !fvec
      %first, %second = tla.load %src_tile {load_dist = #tla.load_dist<dintlv_b32>} : !fsrc -> !tla.vector<64xf32>, !tla.vector<64xf32>
      %encoded = "tla.cast"(%first) <{trait = array<i32: 0, 1, 0>}> : (!tla.vector<64xf32>) -> !tla.vector<256x!tla.hif8>
      %decoded = "tla.cast"(%encoded) <{trait = array<i32: 0, 0, 0>}> : (!tla.vector<256x!tla.hif8>) -> !tla.vector<64xf32>
      tla.store %dst_tile, %decoded : !fvec, !tla.vector<64xf32>
    }) : () -> ()
    return
  }
}

// A direct helper consumer must not leave an AVE load for hivmc to reject.
// CHECK-LABEL: func.func private @vector_region_
// CHECK-NOT: ave.hir.vload
// CHECK: hivm_regbaseintrins.intr.hivm.vldsx2.v64f32
// CHECK: call @lowp_cast_f32_to_hif8_round_sat_slot0
// CHECK: call @lowp_cast_hif8_to_f32_slot0
// CHECK-NOT: ave.hir.vload
// CHECK: return

// CHECK-LABEL: func.func private @vector_region_1
// CHECK: %[[OFFSET:.*]] = llvm.mlir.constant(32 : index) : i64
// CHECK: %[[BASE:.*]] = memref.extract_aligned_pointer_as_index %arg0
// CHECK: %[[BASE64:.*]] = builtin.unrealized_conversion_cast %[[BASE]] : index to i64
// CHECK-NEXT: %[[ADDRESS:.*]] = llvm.add %[[BASE64]], %[[OFFSET]] : i64
// CHECK-NEXT: %[[POINTER:.*]] = llvm.inttoptr %[[ADDRESS]] : i64 to !llvm.ptr<6>
// CHECK-NEXT: %[[PAIR:.*]] = "hivm_regbaseintrins.intr.hivm.vldsx2.v64f32"(%[[POINTER]],
// CHECK-NEXT: %[[FIRST:.*]] = llvm.extractvalue %[[PAIR]][0]
// CHECK-NEXT: %[[ENCODED:.*]] = call @lowp_cast_f32_to_hif8_round_sat_slot0(%[[FIRST]],
// CHECK-NEXT: call @lowp_cast_hif8_to_f32_slot0(%[[ENCODED]],
