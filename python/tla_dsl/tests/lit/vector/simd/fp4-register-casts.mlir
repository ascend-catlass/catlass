// RUN: %tla_compile %s --mlir-print-ir-after=tla-vector-region-simd -o %t 2>&1 | %filecheck %s

// Encodings lower to byte registers; numerical conversion remains a distinct
// always-inline instruction bridge. No staging tensor is introduced.

!s_bf16_to_f4e2m1x2 = !tla.tensor<!tla.layout<!tla.shape<128>, !tla.stride<1>, !tla.shape<128>, RowMajor>, !tla.coord<0>, !tla.ptr<bf16, ub, 32>>
!d_bf16_to_f4e2m1x2 = !tla.tensor<!tla.layout<!tla.shape<256>, !tla.stride<1>, !tla.shape<256>, RowMajor>, !tla.coord<0>, !tla.ptr<!tla.f4e2m1x2, ub, 32>>
!s_f4e2m1x2_to_bf16 = !tla.tensor<!tla.layout<!tla.shape<256>, !tla.stride<1>, !tla.shape<256>, RowMajor>, !tla.coord<0>, !tla.ptr<!tla.f4e2m1x2, ub, 32>>
!d_f4e2m1x2_to_bf16 = !tla.tensor<!tla.layout<!tla.shape<128>, !tla.stride<1>, !tla.shape<128>, RowMajor>, !tla.coord<0>, !tla.ptr<bf16, ub, 32>>
!s_bf16_to_f4e1m2x2 = !tla.tensor<!tla.layout<!tla.shape<128>, !tla.stride<1>, !tla.shape<128>, RowMajor>, !tla.coord<0>, !tla.ptr<bf16, ub, 32>>
!d_bf16_to_f4e1m2x2 = !tla.tensor<!tla.layout<!tla.shape<256>, !tla.stride<1>, !tla.shape<256>, RowMajor>, !tla.coord<0>, !tla.ptr<!tla.f4e1m2x2, ub, 32>>
!s_f4e1m2x2_to_bf16 = !tla.tensor<!tla.layout<!tla.shape<256>, !tla.stride<1>, !tla.shape<256>, RowMajor>, !tla.coord<0>, !tla.ptr<!tla.f4e1m2x2, ub, 32>>
!d_f4e1m2x2_to_bf16 = !tla.tensor<!tla.layout<!tla.shape<128>, !tla.stride<1>, !tla.shape<128>, RowMajor>, !tla.coord<0>, !tla.ptr<bf16, ub, 32>>

module {
  func.func @bf16_to_f4e2m1x2(%sm: memref<128xbf16, #hivm.address_space<ub>>, %dm: memref<256xi8, #hivm.address_space<ub>>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %cs = arith.constant 128 : index
    %cd = arith.constant 256 : index
    %s = tla.tensor_desc %sm shape [%c1, %cs, %c1, %c1] stride [%cs, %c1, %c1, %c1] origin_shape [%c1, %cs] coord [%c0, %c0] : memref<128xbf16, #hivm.address_space<ub>> -> !s_bf16_to_f4e2m1x2
    %d = tla.tensor_desc %dm shape [%c1, %cd, %c1, %c1] stride [%cd, %c1, %c1, %c1] origin_shape [%c1, %cd] coord [%c0, %c0] : memref<256xi8, #hivm.address_space<ub>> -> !d_bf16_to_f4e2m1x2
    "tla.vec.func"() ({
      %v = tla.load %s : !s_bf16_to_f4e2m1x2 -> !tla.vector<128xbf16>
      %r = "tla.cast"(%v) <{trait = array<i32: 2, 1, 4>}> : (!tla.vector<128xbf16>) -> !tla.vector<256x!tla.f4e2m1x2>
      tla.store %d, %r : !d_bf16_to_f4e2m1x2, !tla.vector<256x!tla.f4e2m1x2>
    }) {mode = "simd"} : () -> ()
    return
  }
  func.func @f4e2m1x2_to_bf16(%sm: memref<256xi8, #hivm.address_space<ub>>, %dm: memref<128xbf16, #hivm.address_space<ub>>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %cs = arith.constant 256 : index
    %cd = arith.constant 128 : index
    %s = tla.tensor_desc %sm shape [%c1, %cs, %c1, %c1] stride [%cs, %c1, %c1, %c1] origin_shape [%c1, %cs] coord [%c0, %c0] : memref<256xi8, #hivm.address_space<ub>> -> !s_f4e2m1x2_to_bf16
    %d = tla.tensor_desc %dm shape [%c1, %cd, %c1, %c1] stride [%cd, %c1, %c1, %c1] origin_shape [%c1, %cd] coord [%c0, %c0] : memref<128xbf16, #hivm.address_space<ub>> -> !d_f4e2m1x2_to_bf16
    "tla.vec.func"() ({
      %v = tla.load %s : !s_f4e2m1x2_to_bf16 -> !tla.vector<256x!tla.f4e2m1x2>
      %r = "tla.cast"(%v) <{trait = array<i32: 2, 0, 0>}> : (!tla.vector<256x!tla.f4e2m1x2>) -> !tla.vector<128xbf16>
      tla.store %d, %r : !d_f4e2m1x2_to_bf16, !tla.vector<128xbf16>
    }) {mode = "simd"} : () -> ()
    return
  }
  func.func @bf16_to_f4e1m2x2(%sm: memref<128xbf16, #hivm.address_space<ub>>, %dm: memref<256xi8, #hivm.address_space<ub>>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %cs = arith.constant 128 : index
    %cd = arith.constant 256 : index
    %s = tla.tensor_desc %sm shape [%c1, %cs, %c1, %c1] stride [%cs, %c1, %c1, %c1] origin_shape [%c1, %cs] coord [%c0, %c0] : memref<128xbf16, #hivm.address_space<ub>> -> !s_bf16_to_f4e1m2x2
    %d = tla.tensor_desc %dm shape [%c1, %cd, %c1, %c1] stride [%cd, %c1, %c1, %c1] origin_shape [%c1, %cd] coord [%c0, %c0] : memref<256xi8, #hivm.address_space<ub>> -> !d_bf16_to_f4e1m2x2
    "tla.vec.func"() ({
      %v = tla.load %s : !s_bf16_to_f4e1m2x2 -> !tla.vector<128xbf16>
      %r = "tla.cast"(%v) <{trait = array<i32: 1, 1, 3>}> : (!tla.vector<128xbf16>) -> !tla.vector<256x!tla.f4e1m2x2>
      tla.store %d, %r : !d_bf16_to_f4e1m2x2, !tla.vector<256x!tla.f4e1m2x2>
    }) {mode = "simd"} : () -> ()
    return
  }
  func.func @f4e1m2x2_to_bf16(%sm: memref<256xi8, #hivm.address_space<ub>>, %dm: memref<128xbf16, #hivm.address_space<ub>>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %cs = arith.constant 256 : index
    %cd = arith.constant 128 : index
    %s = tla.tensor_desc %sm shape [%c1, %cs, %c1, %c1] stride [%cs, %c1, %c1, %c1] origin_shape [%c1, %cs] coord [%c0, %c0] : memref<256xi8, #hivm.address_space<ub>> -> !s_f4e1m2x2_to_bf16
    %d = tla.tensor_desc %dm shape [%c1, %cd, %c1, %c1] stride [%cd, %c1, %c1, %c1] origin_shape [%c1, %cd] coord [%c0, %c0] : memref<128xbf16, #hivm.address_space<ub>> -> !d_f4e1m2x2_to_bf16
    "tla.vec.func"() ({
      %v = tla.load %s : !s_f4e1m2x2_to_bf16 -> !tla.vector<256x!tla.f4e1m2x2>
      %r = "tla.cast"(%v) <{trait = array<i32: 1, 0, 0>}> : (!tla.vector<256x!tla.f4e1m2x2>) -> !tla.vector<128xbf16>
      tla.store %d, %r : !d_f4e1m2x2_to_bf16, !tla.vector<128xbf16>
    }) {mode = "simd"} : () -> ()
    return
  }
  func.func @bf16_source_mask_to_f4e2m1x2(%sm: memref<128xbf16, #hivm.address_space<ub>>, %dm: memref<256xi8, #hivm.address_space<ub>>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %cs = arith.constant 128 : index
    %cd = arith.constant 256 : index
    %s = tla.tensor_desc %sm shape [%c1, %cs, %c1, %c1] stride [%cs, %c1, %c1, %c1] origin_shape [%c1, %cs] coord [%c0, %c0] : memref<128xbf16, #hivm.address_space<ub>> -> !s_bf16_to_f4e2m1x2
    %d = tla.tensor_desc %dm shape [%c1, %cd, %c1, %c1] stride [%cd, %c1, %c1, %c1] origin_shape [%c1, %cd] coord [%c0, %c0] : memref<256xi8, #hivm.address_space<ub>> -> !d_bf16_to_f4e2m1x2
    "tla.vec.func"() ({
      %v = tla.load %s : !s_bf16_to_f4e2m1x2 -> !tla.vector<128xbf16>
      %mask = "tla.create_mask"() {pattern = "M3", dtype = bf16} : () -> !tla.mask<128>
      %r = "tla.cast"(%v, %mask) <{trait = array<i32: 0, 0, 5>}> : (!tla.vector<128xbf16>, !tla.mask<128>) -> !tla.vector<256x!tla.f4e2m1x2>
      tla.store %d, %r : !d_bf16_to_f4e2m1x2, !tla.vector<256x!tla.f4e2m1x2>
    }) {mode = "simd"} : () -> ()
    return
  }
  func.func @bf16_source_mask_to_f4e1m2x2(%sm: memref<128xbf16, #hivm.address_space<ub>>, %dm: memref<256xi8, #hivm.address_space<ub>>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %cs = arith.constant 128 : index
    %cd = arith.constant 256 : index
    %s = tla.tensor_desc %sm shape [%c1, %cs, %c1, %c1] stride [%cs, %c1, %c1, %c1] origin_shape [%c1, %cs] coord [%c0, %c0] : memref<128xbf16, #hivm.address_space<ub>> -> !s_bf16_to_f4e1m2x2
    %d = tla.tensor_desc %dm shape [%c1, %cd, %c1, %c1] stride [%cd, %c1, %c1, %c1] origin_shape [%c1, %cd] coord [%c0, %c0] : memref<256xi8, #hivm.address_space<ub>> -> !d_bf16_to_f4e1m2x2
    "tla.vec.func"() ({
      %v = tla.load %s : !s_bf16_to_f4e1m2x2 -> !tla.vector<128xbf16>
      %mask = "tla.create_mask"() {pattern = "M3", dtype = bf16} : () -> !tla.mask<128>
      %r = "tla.cast"(%v, %mask) <{trait = array<i32: 0, 0, 5>}> : (!tla.vector<128xbf16>, !tla.mask<128>) -> !tla.vector<256x!tla.f4e1m2x2>
      tla.store %d, %r : !d_bf16_to_f4e1m2x2, !tla.vector<256x!tla.f4e1m2x2>
    }) {mode = "simd"} : () -> ()
    return
  }
}

// CHECK: call @lowp_cast_bf16_to_f4e2m1x2_rint_sat_slot2({{.*}}) : (vector<128xbf16>, vector<256xi1>) -> vector<256xi8>
// CHECK: call @lowp_cast_f4e2m1x2_to_bf16_slot2({{.*}}) : (vector<256xi8>, vector<256xi1>) -> vector<128xbf16>
// CHECK: call @lowp_cast_bf16_to_f4e1m2x2_trunc_sat_slot1({{.*}}) : (vector<128xbf16>, vector<256xi1>) -> vector<256xi8>
// CHECK: call @lowp_cast_f4e1m2x2_to_bf16_slot1({{.*}}) : (vector<256xi8>, vector<256xi1>) -> vector<128xbf16>
// CHECK: call @lowp_cast_bf16_to_f4e2m1x2_rint_sat_slot0({{.*}}) : (vector<128xbf16>, vector<256xi1>) -> vector<256xi8>
// CHECK: call @lowp_cast_bf16_to_f4e1m2x2_rint_sat_slot0({{.*}}) : (vector<128xbf16>, vector<256xi1>) -> vector<256xi8>
