// RUN: %tla_compile %s -o - --mlir-print-ir-after=tla-vector-region-simd 2>&1 | %filecheck %s

// A packed FP4 pair carrier is one byte, so its 256-lane vector uses the same
// byte-granular load/store and bitwise carriers as i8: no nibble packing, no
// odd-tail preservation, and the predicate selects whole pairs.

// CHECK-DAG: ave.hir.vload
// CHECK-DAG: ave.hir.vnot
// CHECK-DAG: ave.hir.vand
// CHECK-DAG: ave.hir.vor
// CHECK-DAG: ave.hir.vxor
// CHECK-DAG: ave.hir.masked_store
// CHECK-NOT: tla.bitwise_not
// CHECK-NOT: tla.bitwise_and
// CHECK-NOT: tla.bitwise_or
// CHECK-NOT: tla.bitwise_xor

!vec = !tla.tensor<!tla.layout<!tla.shape<256>, !tla.stride<1>, !tla.shape<256>, RowMajor>, !tla.coord<0>, !tla.ptr<!tla.f4e2m1x2, ub, 32>>

module {
  func.func @fp4x2_bitwise_lowering(
      %src0_memref: memref<256xi8, #hivm.address_space<ub>>,
      %src1_memref: memref<256xi8, #hivm.address_space<ub>>,
      %dst_memref: memref<256xi8, #hivm.address_space<ub>>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c256 = arith.constant 256 : index
    %src0 = tla.tensor_desc %src0_memref shape [%c1, %c256, %c1, %c1] stride [%c256, %c1, %c1, %c1] origin_shape [%c1, %c256] coord [%c0, %c0] : memref<256xi8, #hivm.address_space<ub>> -> !vec
    %src1 = tla.tensor_desc %src1_memref shape [%c1, %c256, %c1, %c1] stride [%c256, %c1, %c1, %c1] origin_shape [%c1, %c256] coord [%c0, %c0] : memref<256xi8, #hivm.address_space<ub>> -> !vec
    %dst = tla.tensor_desc %dst_memref shape [%c1, %c256, %c1, %c1] stride [%c256, %c1, %c1, %c1] origin_shape [%c1, %c256] coord [%c0, %c0] : memref<256xi8, #hivm.address_space<ub>> -> !vec
    "tla.vector"() ({
      "tla.vec.func"() ({
        %lhs = "tla.load"(%src0) <{operandSegmentSizes = array<i32: 1, 0, 0>}> : (!vec) -> !tla.vector<256x!tla.f4e2m1x2>
        %rhs = "tla.load"(%src1) <{operandSegmentSizes = array<i32: 1, 0, 0>}> : (!vec) -> !tla.vector<256x!tla.f4e2m1x2>
        %all = "tla.create_mask"() {pattern = "ALL", dtype = i8} : () -> !tla.mask<256>
        %not = "tla.bitwise_not"(%lhs) : (!tla.vector<256x!tla.f4e2m1x2>) -> !tla.vector<256x!tla.f4e2m1x2>
        %and = "tla.bitwise_and"(%lhs, %rhs) : (!tla.vector<256x!tla.f4e2m1x2>, !tla.vector<256x!tla.f4e2m1x2>) -> !tla.vector<256x!tla.f4e2m1x2>
        %or = "tla.bitwise_or"(%and, %rhs, %all) : (!tla.vector<256x!tla.f4e2m1x2>, !tla.vector<256x!tla.f4e2m1x2>, !tla.mask<256>) -> !tla.vector<256x!tla.f4e2m1x2>
        %xor = "tla.bitwise_xor"(%or, %not) : (!tla.vector<256x!tla.f4e2m1x2>, !tla.vector<256x!tla.f4e2m1x2>) -> !tla.vector<256x!tla.f4e2m1x2>
        "tla.store"(%dst, %xor) <{operandSegmentSizes = array<i32: 1, 1, 0, 0, 0>}> : (!vec, !tla.vector<256x!tla.f4e2m1x2>) -> ()
        "tla.store"(%dst, %or, %all) <{operandSegmentSizes = array<i32: 1, 1, 0, 1, 0>}> : (!vec, !tla.vector<256x!tla.f4e2m1x2>, !tla.mask<256>) -> ()
      }) {mode = "simd"} : () -> ()
    }) : () -> ()
    return
  }
}
