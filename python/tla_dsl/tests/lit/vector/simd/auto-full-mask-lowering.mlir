// RUN: %tla_compile %s --mlir-print-ir-after=tla-vector-region-simd -o %t 2>&1 | %filecheck %s

!fvec = !tla.tensor<!tla.layout<!tla.shape<64>, !tla.stride<1>, !tla.shape<64>, RowMajor>, !tla.coord<0>, !tla.ptr<f32, ub, 4>>
!hvec = !tla.tensor<!tla.layout<!tla.shape<128>, !tla.stride<1>, !tla.shape<128>, RowMajor>, !tla.coord<0>, !tla.ptr<f16, ub, 4>>
!bvec = !tla.tensor<!tla.layout<!tla.shape<256>, !tla.stride<1>, !tla.shape<256>, RowMajor>, !tla.coord<0>, !tla.ptr<i8, ub, 4>>
!dvec = !tla.tensor<!tla.layout<!tla.shape<32>, !tla.stride<1>, !tla.shape<32>, RowMajor>, !tla.coord<0>, !tla.ptr<i64, ub, 8>>

module {
  func.func @auto_full_mask(%src: memref<64xf32, #hivm.address_space<ub>>, %dst: memref<64xf32, #hivm.address_space<ub>>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c64 = arith.constant 64 : index
    %c4096 = arith.constant 4096 : index
    "tla.vector"() ({
      "tla.vec.func"() ({
        scf.for %iv = %c0 to %c64 step %c1 {
          %offset = arith.muli %iv, %c64 : index
          %remaining_raw = arith.subi %c4096, %offset : index
          %remaining = arith.minsi %c64, %remaining_raw : index
          %src_tile = tla.tensor_desc %src shape [%c1, %c64, %c1, %c1] stride [%c4096, %c1, %c1, %c1] origin_shape [%c1, %remaining] coord [%c0, %offset] : memref<64xf32, #hivm.address_space<ub>> -> !fvec
          %dst_tile = tla.tensor_desc %dst shape [%c1, %c64, %c1, %c1] stride [%c4096, %c1, %c1, %c1] origin_shape [%c1, %remaining] coord [%c0, %offset] : memref<64xf32, #hivm.address_space<ub>> -> !fvec
          %a = tla.load %src_tile : !fvec -> !tla.vector<64xf32>
          %sum = tla.add %a, %a : !tla.vector<64xf32>, !tla.vector<64xf32> -> !tla.vector<64xf32>
          tla.store %dst_tile, %sum : !fvec, !tla.vector<64xf32>
        }
      }) {mode = "simd"} : () -> ()
    }) : () -> ()
    return
  }

  func.func @default_dynamic_mask(%dst: memref<64xf32, #hivm.address_space<ub>>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c64 = arith.constant 64 : index
    %c65 = arith.constant 65 : index
    %c4096 = arith.constant 4096 : index
    "tla.vector"() ({
      "tla.vec.func"() ({
        scf.for %iv = %c0 to %c65 step %c1 {
          %offset = arith.muli %iv, %c64 : index
          %remaining_raw = arith.subi %c4096, %offset : index
          %remaining = arith.minsi %c64, %remaining_raw : index
          %dst_tile = tla.tensor_desc %dst shape [%c1, %c64, %c1, %c1] stride [%c4096, %c1, %c1, %c1] origin_shape [%c1, %remaining] coord [%c0, %offset] : memref<64xf32, #hivm.address_space<ub>> -> !fvec
          %a = tla.load %dst_tile : !fvec -> !tla.vector<64xf32>
          tla.store %dst_tile, %a : !fvec, !tla.vector<64xf32>
        }
      }) {mode = "simd"} : () -> ()
    }) : () -> ()
    return
  }

  func.func @explicit_tail_mask(%dst: memref<64xf32, #hivm.address_space<ub>>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c64 = arith.constant 64 : index
    %c65 = arith.constant 65 : index
    %c4096 = arith.constant 4096 : index
    "tla.vector"() ({
      "tla.vec.func"() ({
        scf.for %iv = %c0 to %c65 step %c1 {
          %offset = arith.muli %iv, %c64 : index
          %remaining = arith.subi %c4096, %offset : index
          %dst_tile = tla.tensor_desc %dst shape [%c1, %c64, %c1, %c1] stride [%c4096, %c1, %c1, %c1] origin_shape [%c1, %remaining] coord [%c0, %offset] : memref<64xf32, #hivm.address_space<ub>> -> !fvec
          %a = tla.load %dst_tile : !fvec -> !tla.vector<64xf32>
          %tail, %next = tla.update_mask %remaining, f32 : !tla.mask<64>, index
          tla.store %dst_tile, %a mask %tail : !fvec, !tla.vector<64xf32> mask !tla.mask<64>
        }
      }) {mode = "simd"} : () -> ()
    }) : () -> ()
    return
  }

  func.func @auto_full_mask_f16(%dst: memref<128xf16, #hivm.address_space<ub>>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c64 = arith.constant 64 : index
    %c128 = arith.constant 128 : index
    %c8192 = arith.constant 8192 : index
    "tla.vector"() ({
      "tla.vec.func"() ({
        scf.for %iv = %c0 to %c64 step %c1 {
          %offset = arith.muli %iv, %c128 : index
          %remaining = arith.subi %c8192, %offset : index
          %dst_tile = tla.tensor_desc %dst shape [%c1, %c128, %c1, %c1] stride [%c8192, %c1, %c1, %c1] origin_shape [%c1, %remaining] coord [%c0, %offset] : memref<128xf16, #hivm.address_space<ub>> -> !hvec
          %a = tla.load %dst_tile : !hvec -> !tla.vector<128xf16>
          tla.store %dst_tile, %a : !hvec, !tla.vector<128xf16>
        }
      }) {mode = "simd"} : () -> ()
    }) : () -> ()
    return
  }

  func.func @auto_full_mask_i8(%dst: memref<256xi8, #hivm.address_space<ub>>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c64 = arith.constant 64 : index
    %c256 = arith.constant 256 : index
    %c16384 = arith.constant 16384 : index
    "tla.vector"() ({
      "tla.vec.func"() ({
        scf.for %iv = %c0 to %c64 step %c1 {
          %offset = arith.muli %iv, %c256 : index
          %remaining = arith.subi %c16384, %offset : index
          %dst_tile = tla.tensor_desc %dst shape [%c1, %c256, %c1, %c1] stride [%c16384, %c1, %c1, %c1] origin_shape [%c1, %remaining] coord [%c0, %offset] : memref<256xi8, #hivm.address_space<ub>> -> !bvec
          %a = tla.load %dst_tile : !bvec -> !tla.vector<256xi8>
          tla.store %dst_tile, %a : !bvec, !tla.vector<256xi8>
        }
      }) {mode = "simd"} : () -> ()
    }) : () -> ()
    return
  }

  // A full i64 store with an omitted mask writes 64 i32 words under a
  // full B32 predicate.
  func.func @omitted_i64_uses_full_b32_predicate(%dst: memref<32xi64, #hivm.address_space<ub>>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c32 = arith.constant 32 : index
    "tla.vector"() ({
      "tla.vec.func"() ({
        %dst_tile = tla.tensor_desc %dst shape [%c1, %c32, %c1, %c1] stride [%c32, %c1, %c1, %c1] origin_shape [%c1, %c32] coord [%c0, %c0] : memref<32xi64, #hivm.address_space<ub>> -> !dvec
        %a = tla.load %dst_tile : !dvec -> !tla.vector<32xi64>
        tla.store %dst_tile, %a : !dvec, !tla.vector<32xi64>
      }) {mode = "simd"} : () -> ()
    }) : () -> ()
    return
  }

  // Python range lowering commonly carries trip counts through i32 before
  // casting them back to index. A value-preserving cast must not hide the
  // proof that every iteration has a full destination tile.
  func.func @auto_full_mask_index_cast(%dst: memref<64xf32, #hivm.address_space<ub>>) {
    "tla.vector"() ({
      "tla.vec.func"() ({
        %c0 = arith.constant 0 : index
        %c1 = arith.constant 1 : index
        %c64_i32 = arith.constant 64 : i32
        %c4096_i32 = arith.constant 4096 : i32
        %c64 = arith.index_cast %c64_i32 : i32 to index
        %c4096 = arith.index_cast %c4096_i32 : i32 to index
        scf.for %iv = %c0 to %c64 step %c1 {
          %offset = arith.muli %iv, %c64 : index
          %remaining = arith.subi %c4096, %offset : index
          %dst_tile = tla.tensor_desc %dst shape [%c1, %c64, %c1, %c1] stride [%c4096, %c1, %c1, %c1] origin_shape [%c1, %remaining] coord [%c0, %offset] : memref<64xf32, #hivm.address_space<ub>> -> !fvec
          %a = tla.load %dst_tile : !fvec -> !tla.vector<64xf32>
          tla.store %dst_tile, %a : !fvec, !tla.vector<64xf32>
        }
      }) {mode = "simd"} : () -> ()
    }) : () -> ()
    return
  }

  // Overflow makes the interval unknown. It must conservatively retain a
  // predicate instead of proving an unsafe ALL mask.
  func.func @overflow_retains_tail(%dst: memref<64xf32, #hivm.address_space<ub>>) {
    "tla.vector"() ({
      "tla.vec.func"() ({
        %c0 = arith.constant 0 : index
        %c1 = arith.constant 1 : index
        %c64 = arith.constant 64 : index
        %max = arith.constant 9223372036854775807 : index
        %overflow = arith.addi %max, %c1 : index
        %dst_tile = tla.tensor_desc %dst shape [%c1, %c64, %c1, %c1] stride [%c64, %c1, %c1, %c1] origin_shape [%c1, %overflow] coord [%c0, %c0] : memref<64xf32, #hivm.address_space<ub>> -> !fvec
        %a = tla.load %dst_tile : !fvec -> !tla.vector<64xf32>
        tla.store %dst_tile, %a : !fvec, !tla.vector<64xf32>
      }) {mode = "simd"} : () -> ()
    }) : () -> ()
    return
  }

  // i32 arithmetic wraps before index_cast. The mathematical int64 result is
  // positive, but the SSA value is INT32_MIN and must not prove a full tile.
  func.func @narrow_overflow_retains_tail(%dst: memref<64xf32, #hivm.address_space<ub>>) {
    "tla.vector"() ({
      "tla.vec.func"() ({
        %c0 = arith.constant 0 : index
        %c1 = arith.constant 1 : index
        %c2_i32 = arith.constant 2 : i32
        %lower = arith.constant 2147483646 : index
        %upper = arith.constant 2147483647 : index
        %c64 = arith.constant 64 : index
        scf.for %iv = %lower to %upper step %c1 {
          %iv_i32 = arith.index_cast %iv : index to i32
          %wrapped_i32 = arith.addi %iv_i32, %c2_i32 : i32
          %wrapped = arith.index_cast %wrapped_i32 : i32 to index
          %dst_tile = tla.tensor_desc %dst shape [%c1, %c64, %c1, %c1] stride [%c64, %c1, %c1, %c1] origin_shape [%c1, %wrapped] coord [%c0, %c0] : memref<64xf32, #hivm.address_space<ub>> -> !fvec
          %a = tla.load %dst_tile : !fvec -> !tla.vector<64xf32>
          tla.store %dst_tile, %a : !fvec, !tla.vector<64xf32>
        }
      }) {mode = "simd"} : () -> ()
    }) : () -> ()
    return
  }

  // A neighboring i32 computation that stays within range remains provable.
  func.func @narrow_nonoverflow_uses_all(%dst: memref<64xf32, #hivm.address_space<ub>>) {
    "tla.vector"() ({
      "tla.vec.func"() ({
        %c0 = arith.constant 0 : index
        %c1 = arith.constant 1 : index
        %c2_i32 = arith.constant 2 : i32
        %lower = arith.constant 62 : index
        %upper = arith.constant 63 : index
        scf.for %iv = %lower to %upper step %c1 {
          %iv_i32 = arith.index_cast %iv : index to i32
          %c64_i32 = arith.addi %iv_i32, %c2_i32 : i32
          %c64 = arith.index_cast %c64_i32 : i32 to index
          %dst_tile = tla.tensor_desc %dst shape [%c1, %c64, %c1, %c1] stride [%c64, %c1, %c1, %c1] origin_shape [%c1, %c64] coord [%c0, %c0] : memref<64xf32, #hivm.address_space<ub>> -> !fvec
          %a = tla.load %dst_tile : !fvec -> !tla.vector<64xf32>
          tla.store %dst_tile, %a : !fvec, !tla.vector<64xf32>
        }
      }) {mode = "simd"} : () -> ()
    }) : () -> ()
    return
  }

  // Omitted store masks are destination-defined. The cast chain does not alter
  // the cached ALL predicate selected for this full destination tile.
  func.func @auto_full_mask_narrow_widen(%src: memref<64xf32, #hivm.address_space<ub>>, %dst: memref<64xf32, #hivm.address_space<ub>>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c64 = arith.constant 64 : index
    "tla.vector"() ({
      "tla.vec.func"() ({
        %src_tile = tla.tensor_desc %src shape [%c1, %c64, %c1, %c1] stride [%c64, %c1, %c1, %c1] origin_shape [%c1, %c64] coord [%c0, %c0] : memref<64xf32, #hivm.address_space<ub>> -> !fvec
        %dst_tile = tla.tensor_desc %dst shape [%c1, %c64, %c1, %c1] stride [%c64, %c1, %c1, %c1] origin_shape [%c1, %c64] coord [%c0, %c0] : memref<64xf32, #hivm.address_space<ub>> -> !fvec
        %a = tla.load %src_tile : !fvec -> !tla.vector<64xf32>
        %narrow = tla.cast %a, [0, 0, 0] : !tla.vector<64xf32> -> !tla.vector<?xf16>
        %wide = tla.cast %narrow, [0, 0, 0] : !tla.vector<?xf16> -> !tla.vector<?xf32>
        tla.store %dst_tile, %wide : !fvec, !tla.vector<?xf32>
      }) {mode = "simd"} : () -> ()
    }) : () -> ()
    return
  }

}

// CHECK-LABEL: func.func private @vector_region_
// CHECK: ave.hir.pge <ALL> {{.*}} : vector<256xi1>
// CHECK: scf.for
// CHECK: ave.hir.vadd
// CHECK-NOT: ave.hir.plt
// CHECK: ave.hir.masked_store

// CHECK-LABEL: func.func private @vector_region_
// CHECK: scf.for
// CHECK: ave.hir.plt
// CHECK: ave.hir.masked_store

// CHECK-LABEL: func.func private @vector_region_
// CHECK: %[[EXPLICIT_TAIL:.*]], %{{.*}} = ave.hir.plt
// CHECK-NOT: ave.hir.plt
// CHECK: ave.hir.masked_store {{.*}}, %[[EXPLICIT_TAIL]],

// CHECK-LABEL: func.func private @vector_region_
// CHECK: scf.for
// CHECK-NOT: ave.hir.plt
// CHECK: ave.hir.masked_store

// CHECK-LABEL: func.func private @vector_region_
// CHECK: scf.for
// CHECK-NOT: ave.hir.plt
// CHECK: ave.hir.masked_store

// CHECK-LABEL: func.func private @vector_region_
// CHECK: %[[I64_ALL:.*]] = ave.hir.pge <ALL> {functionType = #ave.func_dist_type<pb32>}
// CHECK-NOT: ave.hir.plt
// CHECK: %[[I64_WORDS:.*]] = vector.bitcast {{.*}} : vector<32xi64> to vector<64xi32>
// CHECK-NOT: ave.hir.plt
// CHECK: ave.hir.masked_store <NORM_B32> {{.*}}, %[[I64_ALL]], %[[I64_WORDS]]
// CHECK-SAME: vector<64xi32>

// CHECK-LABEL: func.func private @vector_region_
// CHECK: scf.for
// CHECK-NOT: ave.hir.plt
// CHECK: ave.hir.masked_store

// CHECK-LABEL: func.func private @vector_region_
// CHECK: %[[INDEX_OVERFLOW_PLT:.*]], %{{.*}} = ave.hir.plt
// CHECK: ave.hir.masked_store {{.*}}, %[[INDEX_OVERFLOW_PLT]],

// CHECK-LABEL: func.func private @vector_region_
// CHECK: %[[I32_OVERFLOW_PLT:.*]], %{{.*}} = ave.hir.plt
// CHECK: ave.hir.masked_store {{.*}}, %[[I32_OVERFLOW_PLT]],

// CHECK-LABEL: func.func private @vector_region_
// CHECK: %[[I32_ALL:.*]] = ave.hir.pge <ALL>
// CHECK-NOT: ave.hir.plt
// CHECK: ave.hir.masked_store {{.*}}, %[[I32_ALL]],

// CHECK-LABEL: func.func private @vector_region_
// CHECK: ave.hir.pge <ALL> {{.*}} : vector<256xi1>
// CHECK-NOT: ave.hir.plt
// CHECK: ave.hir.vtruncf
// CHECK: ave.hir.vextf
// CHECK: ave.hir.masked_store
