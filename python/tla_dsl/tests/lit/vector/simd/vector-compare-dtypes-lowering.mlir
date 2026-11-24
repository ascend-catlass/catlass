// RUN: %tla_compile %s --mlir-print-ir-after=tla-vector-region-simd -o %t 2>&1 | %filecheck %s --implicit-check-not=tla.cmp

!i8vec = !tla.tensor<!tla.layout<!tla.shape<256>, !tla.stride<1>, !tla.shape<256>, RowMajor>, !tla.coord<0>, !tla.ptr<i8, ub, 1>>
!bf16vec = !tla.tensor<!tla.layout<!tla.shape<128>, !tla.stride<1>, !tla.shape<128>, RowMajor>, !tla.coord<0>, !tla.ptr<bf16, ub, 2>>
!i64vec = !tla.tensor<!tla.layout<!tla.shape<32>, !tla.stride<1>, !tla.shape<32>, RowMajor>, !tla.coord<0>, !tla.ptr<i64, ub, 8>>
!u8vec = !tla.tensor<!tla.layout<!tla.shape<256>, !tla.stride<1>, !tla.shape<256>, RowMajor>, !tla.coord<0>, !tla.ptr<ui8, ub, 1>>
!u64vec = !tla.tensor<!tla.layout<!tla.shape<32>, !tla.stride<1>, !tla.shape<32>, RowMajor>, !tla.coord<0>, !tla.ptr<ui64, ub, 8>>

module {
  func.func @cmp_i8(
      %lhs_memref: memref<256xi8, #hivm.address_space<ub>>,
      %rhs_memref: memref<256xi8, #hivm.address_space<ub>>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c256 = arith.constant 256 : index
    %lhs = tla.tensor_desc %lhs_memref shape [%c1, %c256, %c1, %c1] stride [%c256, %c1, %c1, %c1] origin_shape [%c1, %c256] coord [%c0, %c0] : memref<256xi8, #hivm.address_space<ub>> -> !i8vec
    %rhs = tla.tensor_desc %rhs_memref shape [%c1, %c256, %c1, %c1] stride [%c256, %c1, %c1, %c1] origin_shape [%c1, %c256] coord [%c0, %c0] : memref<256xi8, #hivm.address_space<ub>> -> !i8vec
    "tla.vec.func"() ({
      %a = tla.load %lhs : !i8vec -> !tla.vector<256xi8>
      %b = tla.load %rhs : !i8vec -> !tla.vector<256xi8>
      %m = tla.cmp "lt" %a, %b : !tla.vector<256xi8>, !tla.vector<256xi8> -> !tla.mask<256>
      tla.store %lhs, %a mask %m : !i8vec, !tla.vector<256xi8> mask !tla.mask<256>
    }) {mode = "simd"} : () -> ()
    return
  }

  func.func @cmp_bf16(
      %lhs_memref: memref<128xbf16, #hivm.address_space<ub>>,
      %rhs_memref: memref<128xbf16, #hivm.address_space<ub>>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c128 = arith.constant 128 : index
    %lhs = tla.tensor_desc %lhs_memref shape [%c1, %c128, %c1, %c1] stride [%c128, %c1, %c1, %c1] origin_shape [%c1, %c128] coord [%c0, %c0] : memref<128xbf16, #hivm.address_space<ub>> -> !bf16vec
    %rhs = tla.tensor_desc %rhs_memref shape [%c1, %c128, %c1, %c1] stride [%c128, %c1, %c1, %c1] origin_shape [%c1, %c128] coord [%c0, %c0] : memref<128xbf16, #hivm.address_space<ub>> -> !bf16vec
    "tla.vec.func"() ({
      %a = tla.load %lhs : !bf16vec -> !tla.vector<128xbf16>
      %b = tla.load %rhs : !bf16vec -> !tla.vector<128xbf16>
      %m = tla.cmp "lt" %a, %b : !tla.vector<128xbf16>, !tla.vector<128xbf16> -> !tla.mask<128>
      tla.store %lhs, %a mask %m : !bf16vec, !tla.vector<128xbf16> mask !tla.mask<128>
    }) {mode = "simd"} : () -> ()
    return
  }

  func.func @cmp_u64(
      %lhs_memref: memref<32xui64, #hivm.address_space<ub>>,
      %rhs_memref: memref<32xui64, #hivm.address_space<ub>>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c32 = arith.constant 32 : index
    %lhs = tla.tensor_desc %lhs_memref shape [%c1, %c32, %c1, %c1] stride [%c32, %c1, %c1, %c1] origin_shape [%c1, %c32] coord [%c0, %c0] : memref<32xui64, #hivm.address_space<ub>> -> !u64vec
    %rhs = tla.tensor_desc %rhs_memref shape [%c1, %c32, %c1, %c1] stride [%c32, %c1, %c1, %c1] origin_shape [%c1, %c32] coord [%c0, %c0] : memref<32xui64, #hivm.address_space<ub>> -> !u64vec
    "tla.vec.func"() ({
      %a = tla.load %lhs : !u64vec -> !tla.vector<32xui64>
      %b = tla.load %rhs : !u64vec -> !tla.vector<32xui64>
      %m = tla.cmp "lt" %a, %b : !tla.vector<32xui64>, !tla.vector<32xui64> -> !tla.mask<32>
      tla.store %lhs, %a mask %m : !u64vec, !tla.vector<32xui64> mask !tla.mask<32>
    }) {mode = "simd"} : () -> ()
    return
  }

  func.func @cmps_u8(
      %lhs_memref: memref<256xui8, #hivm.address_space<ub>>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c256 = arith.constant 256 : index
    %lhs = tla.tensor_desc %lhs_memref shape [%c1, %c256, %c1, %c1] stride [%c256, %c1, %c1, %c1] origin_shape [%c1, %c256] coord [%c0, %c0] : memref<256xui8, #hivm.address_space<ub>> -> !u8vec
    "tla.vec.func"() ({
      %a = tla.load %lhs : !u8vec -> !tla.vector<256xui8>
      %cst = arith.constant 7 : i8
      %m = tla.cmp "lt" %a, %cst : !tla.vector<256xui8>, i8 -> !tla.mask<256>
      tla.store %lhs, %a mask %m : !u8vec, !tla.vector<256xui8> mask !tla.mask<256>
    }) {mode = "simd"} : () -> ()
    return
  }

  func.func @cmps_i64(
      %lhs_memref: memref<32xi64, #hivm.address_space<ub>>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c32 = arith.constant 32 : index
    %lhs = tla.tensor_desc %lhs_memref shape [%c1, %c32, %c1, %c1] stride [%c32, %c1, %c1, %c1] origin_shape [%c1, %c32] coord [%c0, %c0] : memref<32xi64, #hivm.address_space<ub>> -> !i64vec
    "tla.vec.func"() ({
      %a = tla.load %lhs : !i64vec -> !tla.vector<32xi64>
      %cst = arith.constant 7 : i64
      %m = tla.cmp "lt" %a, %cst : !tla.vector<32xi64>, i64 -> !tla.mask<32>
      tla.store %lhs, %a mask %m : !i64vec, !tla.vector<32xi64> mask !tla.mask<32>
    }) {mode = "simd"} : () -> ()
    return
  }

  func.func @cmps_u64(
      %lhs_memref: memref<32xui64, #hivm.address_space<ub>>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c32 = arith.constant 32 : index
    %lhs = tla.tensor_desc %lhs_memref shape [%c1, %c32, %c1, %c1] stride [%c32, %c1, %c1, %c1] origin_shape [%c1, %c32] coord [%c0, %c0] : memref<32xui64, #hivm.address_space<ub>> -> !u64vec
    "tla.vec.func"() ({
      %a = tla.load %lhs : !u64vec -> !tla.vector<32xui64>
      %cst = arith.constant 7 : i64
      %m = tla.cmp "lt" %a, %cst : !tla.vector<32xui64>, i64 -> !tla.mask<32>
      tla.store %lhs, %a mask %m : !u64vec, !tla.vector<32xui64> mask !tla.mask<32>
    }) {mode = "simd"} : () -> ()
    return
  }

  func.func @cmps_u64_runtime(
      %lhs_memref: memref<32xui64, #hivm.address_space<ub>>,
      %threshold: i32) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c32 = arith.constant 32 : index
    %lhs = tla.tensor_desc %lhs_memref shape [%c1, %c32, %c1, %c1] stride [%c32, %c1, %c1, %c1] origin_shape [%c1, %c32] coord [%c0, %c0] : memref<32xui64, #hivm.address_space<ub>> -> !u64vec
    "tla.vec.func"() ({
      %a = tla.load %lhs : !u64vec -> !tla.vector<32xui64>
      %th = arith.extsi %threshold : i32 to i64
      %m = tla.cmp "lt" %a, %th : !tla.vector<32xui64>, i64 -> !tla.mask<32>
      tla.store %lhs, %a mask %m : !u64vec, !tla.vector<32xui64> mask !tla.mask<32>
    }) {mode = "simd"} : () -> ()
    return
  }
}

// CHECK-LABEL: func.func private @vector_region_
// CHECK: ave.hir.vcmp <LT> {{.*}} : vector<256xi8>, vector<256xi1> -> vector<256xi1>
// CHECK-LABEL: func.func private @vector_region_
// CHECK: ave.hir.vcmp <LT> {{.*}} : vector<128xbf16>, vector<256xi1> -> vector<256xi1>
// CHECK-LABEL: func.func private @vector_region_
// CHECK: hivm.punpack
// CHECK: call @cmp_lt_uint64_t({{.*}}) : (vector<32xi64>, vector<32xi64>, vector<256xi1>) -> vector<256xi1>
// CHECK-LABEL: func.func private @vector_region_
// CHECK: ave.hir.vcmps <ULT> {{.*}} : vector<256xi8>, i8, vector<256xi1> -> vector<256xi1>
// CHECK-LABEL: func.func private @vector_region_
// CHECK: call @cmps_lt_int64_t({{.*}}) : (vector<32xi64>, i64, vector<256xi1>) -> vector<256xi1>
// CHECK-LABEL: func.func private @vector_region_
// CHECK: call @cmps_lt_uint64_t({{.*}}) : (vector<32xi64>, i64, vector<256xi1>) -> vector<256xi1>
// CHECK-LABEL: func.func private @vector_region_
// CHECK: arith.extsi {{.*}} : i32 to i64
// CHECK: call @cmps_lt_uint64_t({{.*}}) : (vector<32xi64>, i64, vector<256xi1>) -> vector<256xi1>
