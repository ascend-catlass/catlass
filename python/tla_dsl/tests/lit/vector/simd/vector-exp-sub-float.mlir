// RUN: %tla_compile %s --mlir-print-ir-after=tla-vector-region-simd -o %t 2>&1 | %filecheck %s --implicit-check-not=llvm.hivm.vexpdif

// Both register slots accept f32 inputs and retain all 64 output lanes.
// Native ExpSub supplies ZEROING for the input-domain mask.

!input_vec = !tla.tensor<!tla.layout<!tla.shape<64>, !tla.stride<1>, !tla.shape<64>, RowMajor>, !tla.coord<0>, !tla.ptr<f32, ub, 4>>
!fvec = !tla.tensor<!tla.layout<!tla.shape<64>, !tla.stride<1>, !tla.shape<64>, RowMajor>, !tla.coord<0>, !tla.ptr<f32, ub, 4>>

module {
  func.func @float_exp_sub(
      %lhs_memref: memref<64xf32, #hivm.address_space<ub>>,
      %rhs_memref: memref<64xf32, #hivm.address_space<ub>>,
      %out0_memref: memref<64xf32, #hivm.address_space<ub>>,
      %out1_memref: memref<64xf32, #hivm.address_space<ub>>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %input_lanes = arith.constant 64 : index
    %c64 = arith.constant 64 : index
    %lhs = tla.tensor_desc %lhs_memref shape [%c1, %input_lanes, %c1, %c1] stride [%input_lanes, %c1, %c1, %c1] origin_shape [%c1, %input_lanes] coord [%c0, %c0] : memref<64xf32, #hivm.address_space<ub>> -> !input_vec
    %rhs = tla.tensor_desc %rhs_memref shape [%c1, %input_lanes, %c1, %c1] stride [%input_lanes, %c1, %c1, %c1] origin_shape [%c1, %input_lanes] coord [%c0, %c0] : memref<64xf32, #hivm.address_space<ub>> -> !input_vec
    %out0_dst = tla.tensor_desc %out0_memref shape [%c1, %c64, %c1, %c1] stride [%c64, %c1, %c1, %c1] origin_shape [%c1, %c64] coord [%c0, %c0] : memref<64xf32, #hivm.address_space<ub>> -> !fvec
    %out1_dst = tla.tensor_desc %out1_memref shape [%c1, %c64, %c1, %c1] stride [%c64, %c1, %c1, %c1] origin_shape [%c1, %c64] coord [%c0, %c0] : memref<64xf32, #hivm.address_space<ub>> -> !fvec
    "tla.vec.func"() ({
      %lhs_vec = tla.load %lhs : !input_vec -> !tla.vector<64xf32>
      %rhs_vec = tla.load %rhs : !input_vec -> !tla.vector<64xf32>
      %input_mask = "tla.create_mask"() {pattern = "ALL", dtype = f32} : () -> !tla.mask<64>
      %part0 = tla.exp_sub %lhs_vec, %rhs_vec mask %input_mask part 0 : !tla.vector<64xf32>, !tla.vector<64xf32> mask !tla.mask<64> -> !tla.vector<64xf32>
      %part1 = tla.exp_sub %lhs_vec, %rhs_vec mask %input_mask part 1 : !tla.vector<64xf32>, !tla.vector<64xf32> mask !tla.mask<64> -> !tla.vector<64xf32>
      tla.store %out0_dst, %part0 : !fvec, !tla.vector<64xf32>
      tla.store %out1_dst, %part1 : !fvec, !tla.vector<64xf32>
    }) {mode = "simd"} : () -> ()
    return
  }
}

// CHECK: func.func private @exp_sub_f32(vector<64xf32>, vector<64xf32>, vector<256xi1>) -> vector<64xf32> attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>
// CHECK-NOT: ave.hir.plt
// CHECK-NOT: ave.hir.preg.and
// CHECK: call @exp_sub_f32
// CHECK-NOT: ave.hir.plt
// CHECK-NOT: ave.hir.preg.and
// CHECK: call @exp_sub_f32
// CHECK-NOT: ave.hir.plt
// CHECK-NOT: ave.hir.preg.and
