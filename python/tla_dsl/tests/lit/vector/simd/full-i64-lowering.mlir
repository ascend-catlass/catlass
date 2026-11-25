// RUN: %tla_compile %s --mlir-print-ir-after=tla-vector-region-simd -o %t 2>&1 | %filecheck %s

// 64-bit broadcasts: the AVE 64-bit broadcast ABI cannot legalize a <32xi64>
// result, so both the scalar and the one-lane-fragment forms call linked
// bitcode helpers. PGE has no b64 mode, so an omitted mask is packed with
// punpack before the helper sees the native b64 predicate.

!i64vec = !tla.tensor<!tla.layout<!tla.shape<32>, !tla.stride<1>, !tla.shape<32>, RowMajor>, !tla.coord<0>, !tla.ptr<i64, ub, 8>>
!u64vec = !tla.tensor<!tla.layout<!tla.shape<32>, !tla.stride<1>, !tla.shape<32>, RowMajor>, !tla.coord<0>, !tla.ptr<ui64, ub, 8>>
!i64one = !tla.tensor<!tla.layout<!tla.shape<1>, !tla.stride<1>, !tla.shape<1>, RowMajor>, !tla.coord<0>, !tla.ptr<i64, ub, 8>>

module {
  func.func @full_scalar_i64(%dst: memref<32xi64, #hivm.address_space<ub>>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c32 = arith.constant 32 : index
    %d = tla.tensor_desc %dst shape [%c1, %c32, %c1, %c1] stride [%c32, %c1, %c1, %c1] origin_shape [%c1, %c32] coord [%c0, %c0] : memref<32xi64, #hivm.address_space<ub>> -> !i64vec
    "tla.vec.func"() ({
      %shape = "tla.make_shape"() : () -> !tla.shape<32>
      %coord = "tla.make_coord"() : () -> !tla.coord<0>
      %dt = "tla.tile_view"(%d, %shape, %coord) : (!i64vec, !tla.shape<32>, !tla.coord<0>) -> !i64vec
      %v = arith.constant -1099511627776 : i64
      %full = "tla.full"(%v) : (i64) -> !tla.vector<32xi64>
      "tla.store"(%dt, %full) <{operandSegmentSizes = array<i32: 1, 1, 0, 0, 0>}> : (!i64vec, !tla.vector<32xi64>) -> ()
    }) {mode = "simd"} : () -> ()
    return
  }

  func.func @full_scalar_u64(%dst: memref<32xui64, #hivm.address_space<ub>>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c32 = arith.constant 32 : index
    %d = tla.tensor_desc %dst shape [%c1, %c32, %c1, %c1] stride [%c32, %c1, %c1, %c1] origin_shape [%c1, %c32] coord [%c0, %c0] : memref<32xui64, #hivm.address_space<ub>> -> !u64vec
    "tla.vec.func"() ({
      %shape = "tla.make_shape"() : () -> !tla.shape<32>
      %coord = "tla.make_coord"() : () -> !tla.coord<0>
      %dt = "tla.tile_view"(%d, %shape, %coord) : (!u64vec, !tla.shape<32>, !tla.coord<0>) -> !u64vec
      %v = arith.constant -9223372036854775803 : i64
      %full = "tla.full"(%v) : (i64) -> !tla.vector<32xui64>
      "tla.store"(%dt, %full) <{operandSegmentSizes = array<i32: 1, 1, 0, 0, 0>}> : (!u64vec, !tla.vector<32xui64>) -> ()
    }) {mode = "simd"} : () -> ()
    return
  }

  func.func @full_fragment_i64(%src: memref<1xi64, #hivm.address_space<ub>>, %dst: memref<32xi64, #hivm.address_space<ub>>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c32 = arith.constant 32 : index
    %s = tla.tensor_desc %src shape [%c1, %c1, %c1, %c1] stride [%c1, %c1, %c1, %c1] origin_shape [%c1, %c1] coord [%c0, %c0] : memref<1xi64, #hivm.address_space<ub>> -> !i64one
    %d = tla.tensor_desc %dst shape [%c1, %c32, %c1, %c1] stride [%c32, %c1, %c1, %c1] origin_shape [%c1, %c32] coord [%c0, %c0] : memref<32xi64, #hivm.address_space<ub>> -> !i64vec
    "tla.vec.func"() ({
      %shape = "tla.make_shape"() : () -> !tla.shape<32>
      %coord = "tla.make_coord"() : () -> !tla.coord<0>
      %s1 = "tla.make_shape"() : () -> !tla.shape<1>
      %st = "tla.tile_view"(%s, %s1, %coord) : (!i64one, !tla.shape<1>, !tla.coord<0>) -> !i64one
      %dt = "tla.tile_view"(%d, %shape, %coord) : (!i64vec, !tla.shape<32>, !tla.coord<0>) -> !i64vec
      %v = tla.load %st : !i64one -> !tla.vector<1xi64>
      %full = "tla.full"(%v) : (!tla.vector<1xi64>) -> !tla.vector<32xi64>
      "tla.store"(%dt, %full) <{operandSegmentSizes = array<i32: 1, 1, 0, 0, 0>}> : (!i64vec, !tla.vector<32xi64>) -> ()
    }) {mode = "simd"} : () -> ()
    return
  }
}

// CHECK: func.func private @dup_vec_int64_t(vector<32xi64>, vector<256xi1>) -> vector<32xi64>
// CHECK: func.func private @dup_uint64_t(i64, vector<256xi1>) -> vector<32xi64>
// CHECK: func.func private @dup_int64_t(i64, vector<256xi1>) -> vector<32xi64>
// CHECK-LABEL: func.func private @vector_region_0
// CHECK: {{.*}} = "hivm_regbaseintrins.intr.hivm.punpack"(%{{.*}}, %{{.*}})
// CHECK: call @dup_int64_t(%{{.*}}, %{{.*}}) : (i64, vector<256xi1>) -> vector<32xi64>
// CHECK-LABEL: func.func private @vector_region_1
// CHECK: {{.*}} = "hivm_regbaseintrins.intr.hivm.punpack"(%{{.*}}, %{{.*}})
// CHECK: call @dup_uint64_t(%{{.*}}, %{{.*}}) : (i64, vector<256xi1>) -> vector<32xi64>
// CHECK-LABEL: func.func private @vector_region_2
// CHECK: {{.*}} = "hivm_regbaseintrins.intr.hivm.punpack"(%{{.*}}, %{{.*}})
// CHECK: call @dup_vec_int64_t(%{{.*}}, %{{.*}}) : (vector<32xi64>, vector<256xi1>) -> vector<32xi64>
