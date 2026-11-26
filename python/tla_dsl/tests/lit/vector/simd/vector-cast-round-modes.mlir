// RUN: %tla_compile %s -o - --mlir-print-ir-after=tla-vector-region-simd 2>&1 | %filecheck %s

!f32 = !tla.tensor<!tla.layout<!tla.shape<64>, !tla.stride<1>, !tla.shape<64>, RowMajor>, !tla.coord<0>, !tla.ptr<f32, ub, 4>>
!f16 = !tla.tensor<!tla.layout<!tla.shape<128>, !tla.stride<1>, !tla.shape<128>, RowMajor>, !tla.coord<0>, !tla.ptr<f16, ub, 2>>
!i32 = !tla.tensor<!tla.layout<!tla.shape<64>, !tla.stride<1>, !tla.shape<64>, RowMajor>, !tla.coord<0>, !tla.ptr<i32, ub, 4>>

// CAST_NONE (4) and CAST_RINT (5) both lower to RINT when the cast rounds.
// CHECK-LABEL: func.func @f32_rounding(
// CHECK: ave.hir.vtrc %{{.*}}, <trunc>, %{{.*}} : vector<64xf32>, vector<256xi1>, vector<64xf32>
// CHECK-COUNT-2: ave.hir.vtrc %{{.*}}, <rint>, %{{.*}} : vector<64xf32>, vector<256xi1>, vector<64xf32>
// CHECK: ave.hir.vtruncf {{.*}}<odd>{{.*}} : vector<64xf32>, vector<128xf16>
// CHECK: ave.hir.vtruncf {{.*}}<rint>{{.*}} : vector<64xf32>, vector<128xf16>
// CHECK-COUNT-2: ave.hir.vfptosi {{.*}}<rint>{{.*}} : vector<64xf32>, vector<256xi1>, vector<64xi32>
func.func @f32_rounding(
    %input: memref<64xf32, #hivm.address_space<ub>>,
    %f32_trunc: memref<64xf32, #hivm.address_space<ub>>,
    %f32_rint: memref<64xf32, #hivm.address_space<ub>>,
    %f32_default: memref<64xf32, #hivm.address_space<ub>>,
    %f16_odd: memref<128xf16, #hivm.address_space<ub>>,
    %f16_default: memref<128xf16, #hivm.address_space<ub>>,
    %i32_rint: memref<64xi32, #hivm.address_space<ub>>,
    %i32_default: memref<64xi32, #hivm.address_space<ub>>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c64 = arith.constant 64 : index
  %c128 = arith.constant 128 : index
  %src = tla.tensor_desc %input shape [%c1, %c64, %c1, %c1] stride [%c64, %c1, %c1, %c1] origin_shape [%c1, %c64] coord [%c0, %c0] : memref<64xf32, #hivm.address_space<ub>> -> !f32
  %out_f32_trunc = tla.tensor_desc %f32_trunc shape [%c1, %c64, %c1, %c1] stride [%c64, %c1, %c1, %c1] origin_shape [%c1, %c64] coord [%c0, %c0] : memref<64xf32, #hivm.address_space<ub>> -> !f32
  %out_f32_rint = tla.tensor_desc %f32_rint shape [%c1, %c64, %c1, %c1] stride [%c64, %c1, %c1, %c1] origin_shape [%c1, %c64] coord [%c0, %c0] : memref<64xf32, #hivm.address_space<ub>> -> !f32
  %out_f32_default = tla.tensor_desc %f32_default shape [%c1, %c64, %c1, %c1] stride [%c64, %c1, %c1, %c1] origin_shape [%c1, %c64] coord [%c0, %c0] : memref<64xf32, #hivm.address_space<ub>> -> !f32
  %out_f16_odd = tla.tensor_desc %f16_odd shape [%c1, %c128, %c1, %c1] stride [%c128, %c1, %c1, %c1] origin_shape [%c1, %c128] coord [%c0, %c0] : memref<128xf16, #hivm.address_space<ub>> -> !f16
  %out_f16_default = tla.tensor_desc %f16_default shape [%c1, %c128, %c1, %c1] stride [%c128, %c1, %c1, %c1] origin_shape [%c1, %c128] coord [%c0, %c0] : memref<128xf16, #hivm.address_space<ub>> -> !f16
  %out_i32_rint = tla.tensor_desc %i32_rint shape [%c1, %c64, %c1, %c1] stride [%c64, %c1, %c1, %c1] origin_shape [%c1, %c64] coord [%c0, %c0] : memref<64xi32, #hivm.address_space<ub>> -> !i32
  %out_i32_default = tla.tensor_desc %i32_default shape [%c1, %c64, %c1, %c1] stride [%c64, %c1, %c1, %c1] origin_shape [%c1, %c64] coord [%c0, %c0] : memref<64xi32, #hivm.address_space<ub>> -> !i32
  "tla.vec.func"() ({
    %value = tla.load %src : !f32 -> !tla.vector<64xf32>
    %trunc = tla.cast %value, [0, 2, 3] : !tla.vector<64xf32> -> !tla.vector<64xf32>
    %f32_even = tla.cast %value, [0, 2, 5] : !tla.vector<64xf32> -> !tla.vector<64xf32>
    %f32_none = tla.cast %value, [0, 2, 4] : !tla.vector<64xf32> -> !tla.vector<64xf32>
    %odd = tla.cast %value, [0, 2, 6] : !tla.vector<64xf32> -> !tla.vector<128xf16>
    %f16_none = tla.cast %value, [0, 2, 4] : !tla.vector<64xf32> -> !tla.vector<128xf16>
    %i32_even = tla.cast %value, [0, 2, 5] : !tla.vector<64xf32> -> !tla.vector<64xi32>
    %i32_none = tla.cast %value, [0, 2, 4] : !tla.vector<64xf32> -> !tla.vector<64xi32>
    tla.store %out_f32_trunc, %trunc : !f32, !tla.vector<64xf32>
    tla.store %out_f32_rint, %f32_even : !f32, !tla.vector<64xf32>
    tla.store %out_f32_default, %f32_none : !f32, !tla.vector<64xf32>
    tla.store %out_f16_odd, %odd : !f16, !tla.vector<128xf16>
    tla.store %out_f16_default, %f16_none : !f16, !tla.vector<128xf16>
    tla.store %out_i32_rint, %i32_even : !i32, !tla.vector<64xi32>
    tla.store %out_i32_default, %i32_none : !i32, !tla.vector<64xi32>
  }) {mode = "simd"} : () -> ()
  return
}

// CHECK-LABEL: func.func @f16_widen(
// CHECK-NOT: ave.hir.vtrc
// CHECK: ave.hir.vextf %{{.*}}, <part_even>, %{{.*}} : vector<128xf16>, vector<64xf32>, vector<256xi1>
// CHECK-NOT: ave.hir.vtrc
func.func @f16_widen(
    %input: memref<128xf16, #hivm.address_space<ub>>,
    %output: memref<64xf32, #hivm.address_space<ub>>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c64 = arith.constant 64 : index
  %c128 = arith.constant 128 : index
  %src = tla.tensor_desc %input shape [%c1, %c128, %c1, %c1] stride [%c128, %c1, %c1, %c1] origin_shape [%c1, %c128] coord [%c0, %c0] : memref<128xf16, #hivm.address_space<ub>> -> !f16
  %dst = tla.tensor_desc %output shape [%c1, %c64, %c1, %c1] stride [%c64, %c1, %c1, %c1] origin_shape [%c1, %c64] coord [%c0, %c0] : memref<64xf32, #hivm.address_space<ub>> -> !f32
  "tla.vec.func"() ({
    %value = tla.load %src : !f16 -> !tla.vector<128xf16>
    %wide = tla.cast %value, [0, 2, 4] : !tla.vector<128xf16> -> !tla.vector<64xf32>
    tla.store %dst, %wide : !f32, !tla.vector<64xf32>
  }) {mode = "simd"} : () -> ()
  return
}
