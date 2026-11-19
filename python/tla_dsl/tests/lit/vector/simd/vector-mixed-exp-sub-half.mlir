// RUN: %tla_compile %s --mlir-print-ir-after=tla-vector-region-simd -o %t 2>&1 | %filecheck %s --implicit-check-not=llvm.hivm.vexpdif

// The mixed instruction consumes half-domain predicates and produces a full
// float32 register. Both parts must be present; a float-domain predicate
// would silently zero the odd part on hardware.

!hvec = !tla.tensor<!tla.layout<!tla.shape<128>, !tla.stride<1>, !tla.shape<128>, RowMajor>, !tla.coord<0>, !tla.ptr<f16, ub, 2>>
!fvec = !tla.tensor<!tla.layout<!tla.shape<64>, !tla.stride<1>, !tla.shape<64>, RowMajor>, !tla.coord<0>, !tla.ptr<f32, ub, 4>>

module {
  func.func @mixed_exp_sub_half(
      %lhs_memref: memref<128xf16, #hivm.address_space<ub>>,
      %rhs_memref: memref<128xf16, #hivm.address_space<ub>>,
      %even_memref: memref<64xf32, #hivm.address_space<ub>>,
      %odd_memref: memref<64xf32, #hivm.address_space<ub>>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c128 = arith.constant 128 : index
    %c64 = arith.constant 64 : index
    %lhs = tla.tensor_desc %lhs_memref shape [%c1, %c128, %c1, %c1] stride [%c128, %c1, %c1, %c1] origin_shape [%c1, %c128] coord [%c0, %c0] : memref<128xf16, #hivm.address_space<ub>> -> !hvec
    %rhs = tla.tensor_desc %rhs_memref shape [%c1, %c128, %c1, %c1] stride [%c128, %c1, %c1, %c1] origin_shape [%c1, %c128] coord [%c0, %c0] : memref<128xf16, #hivm.address_space<ub>> -> !hvec
    %even_dst = tla.tensor_desc %even_memref shape [%c1, %c64, %c1, %c1] stride [%c64, %c1, %c1, %c1] origin_shape [%c1, %c64] coord [%c0, %c0] : memref<64xf32, #hivm.address_space<ub>> -> !fvec
    %odd_dst = tla.tensor_desc %odd_memref shape [%c1, %c64, %c1, %c1] stride [%c64, %c1, %c1, %c1] origin_shape [%c1, %c64] coord [%c0, %c0] : memref<64xf32, #hivm.address_space<ub>> -> !fvec
    "tla.vec.func"() ({
      %lhs_vec = tla.load %lhs : !hvec -> !tla.vector<128xf16>
      %rhs_vec = tla.load %rhs : !hvec -> !tla.vector<128xf16>
      %half_mask = "tla.create_mask"() {pattern = "ALL", dtype = f16} : () -> !tla.mask<128>
      %even = tla.exp_sub %lhs_vec, %rhs_vec mask %half_mask part 0 : !tla.vector<128xf16>, !tla.vector<128xf16> mask !tla.mask<128> -> !tla.vector<64xf32>
      %odd = tla.exp_sub %lhs_vec, %rhs_vec mask %half_mask part 1 : !tla.vector<128xf16>, !tla.vector<128xf16> mask !tla.mask<128> -> !tla.vector<64xf32>
      tla.store %even_dst, %even : !fvec, !tla.vector<64xf32>
      tla.store %odd_dst, %odd : !fvec, !tla.vector<64xf32>
    }) {mode = "simd"} : () -> ()
    return
  }
}

// CHECK-DAG: func.func private @exp_sub_f16_even(vector<128xf16>, vector<128xf16>, vector<256xi1>) -> vector<64xf32> attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>
// CHECK-DAG: func.func private @exp_sub_f16_odd(vector<128xf16>, vector<128xf16>, vector<256xi1>) -> vector<64xf32> attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>
// CHECK-NOT: ave.hir.plt
// CHECK-NOT: ave.hir.preg.and
// CHECK: call @exp_sub_f16_even
// CHECK-NOT: ave.hir.plt
// CHECK-NOT: ave.hir.preg.and
// CHECK: call @exp_sub_f16_odd
// CHECK-NOT: ave.hir.plt
// CHECK-NOT: ave.hir.preg.and
