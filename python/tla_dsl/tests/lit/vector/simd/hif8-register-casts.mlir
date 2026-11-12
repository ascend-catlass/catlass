// RUN: %tla_compile %s --mlir-print-ir-after=tla-vector-region-simd -o %t 2>&1 | %filecheck %s
// RUN: sed -e 's/array<i32: 0, 1, 0>/array<i32: 2, 2, 0>/' -e 's/array<i32: 3, 0, 0>/array<i32: 1, 0, 0>/' %s > %t.slots.mlir
// RUN: %tla_compile %t.slots.mlir --mlir-print-ir-after=tla-vector-region-simd -o %t.slots 2>&1 | %filecheck %s --check-prefix=ALTERNATE

// RUN: sed 's/array<i32: 0, 1, 0>/array<i32: 0, 1, 7>/' %s > %t.hybrid.mlir
// RUN: %tla_compile %t.hybrid.mlir --mlir-print-ir-after=tla-vector-region-simd -o %t.hybrid 2>&1 | %filecheck %s --check-prefix=HYBRID

// Encodings lower to byte registers; numerical conversion remains a distinct
// always-inline instruction bridge. No staging tensor is introduced.

!s_f32_to_hif8 = !tla.tensor<!tla.layout<!tla.shape<64>, !tla.stride<1>, !tla.shape<64>, RowMajor>, !tla.coord<0>, !tla.ptr<f32, ub, 32>>
!d_f32_to_hif8 = !tla.tensor<!tla.layout<!tla.shape<256>, !tla.stride<1>, !tla.shape<256>, RowMajor>, !tla.coord<0>, !tla.ptr<!tla.hif8, ub, 32>>
!s_hif8_to_f32 = !tla.tensor<!tla.layout<!tla.shape<256>, !tla.stride<1>, !tla.shape<256>, RowMajor>, !tla.coord<0>, !tla.ptr<!tla.hif8, ub, 32>>
!d_hif8_to_f32 = !tla.tensor<!tla.layout<!tla.shape<64>, !tla.stride<1>, !tla.shape<64>, RowMajor>, !tla.coord<0>, !tla.ptr<f32, ub, 32>>

module {
  func.func @f32_to_hif8(%sm: memref<64xf32, #hivm.address_space<ub>>, %dm: memref<256xi8, #hivm.address_space<ub>>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %cs = arith.constant 64 : index
    %cd = arith.constant 256 : index
    %s = tla.tensor_desc %sm shape [%c1, %cs, %c1, %c1] stride [%cs, %c1, %c1, %c1] origin_shape [%c1, %cs] coord [%c0, %c0] : memref<64xf32, #hivm.address_space<ub>> -> !s_f32_to_hif8
    %d = tla.tensor_desc %dm shape [%c1, %cd, %c1, %c1] stride [%cd, %c1, %c1, %c1] origin_shape [%c1, %cd] coord [%c0, %c0] : memref<256xi8, #hivm.address_space<ub>> -> !d_f32_to_hif8
    "tla.vec.func"() ({
      %v = tla.load %s : !s_f32_to_hif8 -> !tla.vector<64xf32>
      %r = "tla.cast"(%v) <{trait = array<i32: 0, 1, 0>}> : (!tla.vector<64xf32>) -> !tla.vector<256x!tla.hif8>
      tla.store %d, %r : !d_f32_to_hif8, !tla.vector<256x!tla.hif8>
    }) {mode = "simd"} : () -> ()
    return
  }
  func.func @hif8_to_f32(%sm: memref<256xi8, #hivm.address_space<ub>>, %dm: memref<64xf32, #hivm.address_space<ub>>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %cs = arith.constant 256 : index
    %cd = arith.constant 64 : index
    %s = tla.tensor_desc %sm shape [%c1, %cs, %c1, %c1] stride [%cs, %c1, %c1, %c1] origin_shape [%c1, %cs] coord [%c0, %c0] : memref<256xi8, #hivm.address_space<ub>> -> !s_hif8_to_f32
    %d = tla.tensor_desc %dm shape [%c1, %cd, %c1, %c1] stride [%cd, %c1, %c1, %c1] origin_shape [%c1, %cd] coord [%c0, %c0] : memref<64xf32, #hivm.address_space<ub>> -> !d_hif8_to_f32
    "tla.vec.func"() ({
      %v = tla.load %s : !s_hif8_to_f32 -> !tla.vector<256x!tla.hif8>
      %r = "tla.cast"(%v) <{trait = array<i32: 3, 0, 0>}> : (!tla.vector<256x!tla.hif8>) -> !tla.vector<64xf32>
      tla.store %d, %r : !d_hif8_to_f32, !tla.vector<64xf32>
    }) {mode = "simd"} : () -> ()
    return
  }
}

// CHECK-DAG: func.func private @lowp_cast_f32_to_hif8_round_sat_slot0(vector<64xf32>, vector<256xi1>) -> vector<256xi8> attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
// CHECK-DAG: func.func private @lowp_cast_hif8_to_f32_slot3(vector<256xi8>, vector<256xi1>) -> vector<64xf32> attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
// CHECK-LABEL: func.func private @vector_region_0
// CHECK: call @lowp_cast_f32_to_hif8_round_sat_slot0({{.*}}) : (vector<64xf32>, vector<256xi1>) -> vector<256xi8>
// CHECK-LABEL: func.func private @vector_region_1
// CHECK: call @lowp_cast_hif8_to_f32_slot3({{.*}}) : (vector<256xi8>, vector<256xi1>) -> vector<64xf32>

// ALTERNATE-LABEL: func.func private @vector_region_0
// ALTERNATE: call @lowp_cast_f32_to_hif8_round_nosat_slot2({{.*}}) : (vector<64xf32>, vector<256xi1>) -> vector<256xi8>
// ALTERNATE-LABEL: func.func private @vector_region_1
// ALTERNATE: call @lowp_cast_hif8_to_f32_slot1({{.*}}) : (vector<256xi8>, vector<256xi1>) -> vector<64xf32>

// HYBRID-LABEL: func.func private @vector_region_0
// HYBRID: call @lowp_cast_f32_to_hif8_hybrid_sat_slot0({{.*}}) : (vector<64xf32>, vector<256xi1>) -> vector<256xi8>
