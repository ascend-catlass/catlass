// RUN: %tla_compile %s -o - --mlir-print-ir-after=tla-vector-region-simd 2>&1 | %filecheck %s
// RUN: sed '/256xf8E4M3FN>.*-> !tla.vector<128xf16>/s/array<i32: 0, 0, 4>/array<i32: 0, 1, 4>/' %s > %t.sat-fp8.mlir
// RUN: not %tla_compile %t.sat-fp8.mlir -o %t 2>&1 | %filecheck %s --check-prefix=SAT-FP8
// SAT-FP8: 'f8E4M3FN'->'f16' requires CAST_NONE and SatMode.UNKNOWN
// RUN: sed '/64xf32>).*-> !tla.vector<256xf8E4M3FN>/s/array<i32: 0, 2, 5>/array<i32: 0, 2, 0>/' %s > %t.f32-round.mlir
// RUN: not %tla_compile %t.f32-round.mlir -o %t 2>&1 | %filecheck %s --check-prefix=ROUND-F32
// RUN: sed '/128xf16>.*-> !tla.vector<256xf8E4M3FN>/s/array<i32: 1, 2, 4>/array<i32: 1, 2, 3>/' %s > %t.f16-trunc.mlir
// RUN: not %tla_compile %t.f16-trunc.mlir -o %t 2>&1 | %filecheck %s --check-prefix=ROUND-F16
// ROUND-F32: {{.*}}f32{{.*}}->{{.*}}f8E4M3FN{{.*}} requires CAST_RINT or CAST_NONE
// ROUND-F16: {{.*}}f16{{.*}}->{{.*}}f8E4M3FN{{.*}} requires CAST_RINT or CAST_NONE

// FP8 <-> f32 uses a 4x byte selector (pp0 through pp3). A conversion
// through f32 uses two 64-lane vectors for the even and odd 16-bit lanes.
// On the FP8 leg, pp = 2 * half + slotBit, with half = 0 for even lanes
// and half = 1 for odd lanes. Slot 0 uses pp0/pp2 and slot 1 uses pp1/pp3.
// Using the even/odd part directly as pp would silently skip half the lanes.

!s_e4m3_to_f32 = !tla.tensor<!tla.layout<!tla.shape<256>, !tla.stride<1>, !tla.shape<256>, RowMajor>, !tla.coord<0>, !tla.ptr<f8E4M3FN, ub, 1>>
!d_e4m3_to_f32 = !tla.tensor<!tla.layout<!tla.shape<64>, !tla.stride<1>, !tla.shape<64>, RowMajor>, !tla.coord<0>, !tla.ptr<f32, ub, 4>>
!s_f32_to_e4m3 = !tla.tensor<!tla.layout<!tla.shape<64>, !tla.stride<1>, !tla.shape<64>, RowMajor>, !tla.coord<0>, !tla.ptr<f32, ub, 4>>
!d_f32_to_e4m3 = !tla.tensor<!tla.layout<!tla.shape<256>, !tla.stride<1>, !tla.shape<256>, RowMajor>, !tla.coord<0>, !tla.ptr<f8E4M3FN, ub, 1>>
!s_e5m2_to_f32 = !tla.tensor<!tla.layout<!tla.shape<256>, !tla.stride<1>, !tla.shape<256>, RowMajor>, !tla.coord<0>, !tla.ptr<f8E5M2, ub, 1>>
!d_e5m2_to_f32 = !tla.tensor<!tla.layout<!tla.shape<64>, !tla.stride<1>, !tla.shape<64>, RowMajor>, !tla.coord<0>, !tla.ptr<f32, ub, 4>>
!s_f32_to_e5m2 = !tla.tensor<!tla.layout<!tla.shape<64>, !tla.stride<1>, !tla.shape<64>, RowMajor>, !tla.coord<0>, !tla.ptr<f32, ub, 4>>
!d_f32_to_e5m2 = !tla.tensor<!tla.layout<!tla.shape<256>, !tla.stride<1>, !tla.shape<256>, RowMajor>, !tla.coord<0>, !tla.ptr<f8E5M2, ub, 1>>
!s_e5m2_to_f16 = !tla.tensor<!tla.layout<!tla.shape<256>, !tla.stride<1>, !tla.shape<256>, RowMajor>, !tla.coord<0>, !tla.ptr<f8E5M2, ub, 1>>
!d_e5m2_to_f16 = !tla.tensor<!tla.layout<!tla.shape<128>, !tla.stride<1>, !tla.shape<128>, RowMajor>, !tla.coord<0>, !tla.ptr<f16, ub, 2>>

!s_e4m3_to_f16 = !tla.tensor<!tla.layout<!tla.shape<256>, !tla.stride<1>, !tla.shape<256>, RowMajor>, !tla.coord<0>, !tla.ptr<f8E4M3FN, ub, 1>>
!d_e4m3_to_f16 = !tla.tensor<!tla.layout<!tla.shape<128>, !tla.stride<1>, !tla.shape<128>, RowMajor>, !tla.coord<0>, !tla.ptr<f16, ub, 2>>
!s_f16_to_e4m3 = !tla.tensor<!tla.layout<!tla.shape<128>, !tla.stride<1>, !tla.shape<128>, RowMajor>, !tla.coord<0>, !tla.ptr<f16, ub, 2>>
!d_f16_to_e4m3 = !tla.tensor<!tla.layout<!tla.shape<256>, !tla.stride<1>, !tla.shape<256>, RowMajor>, !tla.coord<0>, !tla.ptr<f8E4M3FN, ub, 1>>
!s_e4m3_to_bf16 = !tla.tensor<!tla.layout<!tla.shape<256>, !tla.stride<1>, !tla.shape<256>, RowMajor>, !tla.coord<0>, !tla.ptr<f8E4M3FN, ub, 1>>
!d_e4m3_to_bf16 = !tla.tensor<!tla.layout<!tla.shape<128>, !tla.stride<1>, !tla.shape<128>, RowMajor>, !tla.coord<0>, !tla.ptr<bf16, ub, 2>>
!s_bf16_to_e4m3 = !tla.tensor<!tla.layout<!tla.shape<128>, !tla.stride<1>, !tla.shape<128>, RowMajor>, !tla.coord<0>, !tla.ptr<bf16, ub, 2>>
!d_bf16_to_e4m3 = !tla.tensor<!tla.layout<!tla.shape<256>, !tla.stride<1>, !tla.shape<256>, RowMajor>, !tla.coord<0>, !tla.ptr<f8E4M3FN, ub, 1>>
!s_f16_to_e5m2 = !tla.tensor<!tla.layout<!tla.shape<128>, !tla.stride<1>, !tla.shape<128>, RowMajor>, !tla.coord<0>, !tla.ptr<f16, ub, 2>>
!d_f16_to_e5m2 = !tla.tensor<!tla.layout<!tla.shape<256>, !tla.stride<1>, !tla.shape<256>, RowMajor>, !tla.coord<0>, !tla.ptr<f8E5M2, ub, 1>>
!s_e5m2_to_bf16 = !tla.tensor<!tla.layout<!tla.shape<256>, !tla.stride<1>, !tla.shape<256>, RowMajor>, !tla.coord<0>, !tla.ptr<f8E5M2, ub, 1>>
!d_e5m2_to_bf16 = !tla.tensor<!tla.layout<!tla.shape<128>, !tla.stride<1>, !tla.shape<128>, RowMajor>, !tla.coord<0>, !tla.ptr<bf16, ub, 2>>
!s_bf16_to_e5m2 = !tla.tensor<!tla.layout<!tla.shape<128>, !tla.stride<1>, !tla.shape<128>, RowMajor>, !tla.coord<0>, !tla.ptr<bf16, ub, 2>>
!d_bf16_to_e5m2 = !tla.tensor<!tla.layout<!tla.shape<256>, !tla.stride<1>, !tla.shape<256>, RowMajor>, !tla.coord<0>, !tla.ptr<f8E5M2, ub, 1>>

module {
// CHECK-LABEL: func.func @e4m3_to_f32
// CHECK: "hivm_regbaseintrins.intr.hivm.vcvtff.f8e4m32f32.x"
// CHECK-NOT: ave.hir.vextf
  func.func @e4m3_to_f32(
      %sm: memref<256xf8E4M3FN, #hivm.address_space<ub>>,
      %dm: memref<64xf32, #hivm.address_space<ub>>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %cs = arith.constant 256 : index
    %cd = arith.constant 64 : index
    %s = tla.tensor_desc %sm shape [%c1, %cs, %c1, %c1] stride [%cs, %c1, %c1, %c1] origin_shape [%c1, %cs] coord [%c0, %c0] : memref<256xf8E4M3FN, #hivm.address_space<ub>> -> !s_e4m3_to_f32
    %d = tla.tensor_desc %dm shape [%c1, %cd, %c1, %c1] stride [%cd, %c1, %c1, %c1] origin_shape [%c1, %cd] coord [%c0, %c0] : memref<64xf32, #hivm.address_space<ub>> -> !d_e4m3_to_f32
    "tla.vector"() ({
      "tla.vec.func"() ({
        %ss = "tla.make_shape"() : () -> !tla.shape<256>
        %ds = "tla.make_shape"() : () -> !tla.shape<64>
        %co = "tla.make_coord"() : () -> !tla.coord<0>
        %st = "tla.tile_view"(%s, %ss, %co) : (!s_e4m3_to_f32, !tla.shape<256>, !tla.coord<0>) -> !s_e4m3_to_f32
        %dt = "tla.tile_view"(%d, %ds, %co) : (!d_e4m3_to_f32, !tla.shape<64>, !tla.coord<0>) -> !d_e4m3_to_f32
        %r = "tla.load"(%st) : (!s_e4m3_to_f32) -> !tla.vector<256xf8E4M3FN>
        // trait = [reg_slot, sat_mode, round_mode]
        %o = "tla.cast"(%r) <{trait = array<i32: 0, 0, 4>}> : (!tla.vector<256xf8E4M3FN>) -> !tla.vector<64xf32>
        "tla.store"(%dt, %o) <{operandSegmentSizes = array<i32: 1, 1, 0, 0, 0>}> : (!d_e4m3_to_f32, !tla.vector<64xf32>) -> ()
      }) {mode = "simd"} : () -> ()
    }) : () -> ()
    return
  }

// CHECK-LABEL: func.func @f32_to_e4m3
// CHECK: "hivm_regbaseintrins.intr.hivm.vcvtff.f322f8e4m3.x"
// CHECK-NOT: ave.hir.vtruncf
  func.func @f32_to_e4m3(
      %sm: memref<64xf32, #hivm.address_space<ub>>,
      %dm: memref<256xf8E4M3FN, #hivm.address_space<ub>>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %cs = arith.constant 64 : index
    %cd = arith.constant 256 : index
    %s = tla.tensor_desc %sm shape [%c1, %cs, %c1, %c1] stride [%cs, %c1, %c1, %c1] origin_shape [%c1, %cs] coord [%c0, %c0] : memref<64xf32, #hivm.address_space<ub>> -> !s_f32_to_e4m3
    %d = tla.tensor_desc %dm shape [%c1, %cd, %c1, %c1] stride [%cd, %c1, %c1, %c1] origin_shape [%c1, %cd] coord [%c0, %c0] : memref<256xf8E4M3FN, #hivm.address_space<ub>> -> !d_f32_to_e4m3
    "tla.vector"() ({
      "tla.vec.func"() ({
        %ss = "tla.make_shape"() : () -> !tla.shape<64>
        %ds = "tla.make_shape"() : () -> !tla.shape<256>
        %co = "tla.make_coord"() : () -> !tla.coord<0>
        %st = "tla.tile_view"(%s, %ss, %co) : (!s_f32_to_e4m3, !tla.shape<64>, !tla.coord<0>) -> !s_f32_to_e4m3
        %dt = "tla.tile_view"(%d, %ds, %co) : (!d_f32_to_e4m3, !tla.shape<256>, !tla.coord<0>) -> !d_f32_to_e4m3
        %r = "tla.load"(%st) : (!s_f32_to_e4m3) -> !tla.vector<64xf32>
        // trait = [reg_slot, sat_mode, round_mode]
        %o = "tla.cast"(%r) <{trait = array<i32: 0, 2, 5>}> : (!tla.vector<64xf32>) -> !tla.vector<256xf8E4M3FN>
        "tla.store"(%dt, %o) <{operandSegmentSizes = array<i32: 1, 1, 0, 0, 0>}> : (!d_f32_to_e4m3, !tla.vector<256xf8E4M3FN>) -> ()
      }) {mode = "simd"} : () -> ()
    }) : () -> ()
    return
  }

// CHECK-LABEL: func.func @e5m2_to_f32
// CHECK: "hivm_regbaseintrins.intr.hivm.vcvtff.f8e5m22f32.x"
// CHECK-NOT: ave.hir.vextf
  func.func @e5m2_to_f32(
      %sm: memref<256xf8E5M2, #hivm.address_space<ub>>,
      %dm: memref<64xf32, #hivm.address_space<ub>>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %cs = arith.constant 256 : index
    %cd = arith.constant 64 : index
    %s = tla.tensor_desc %sm shape [%c1, %cs, %c1, %c1] stride [%cs, %c1, %c1, %c1] origin_shape [%c1, %cs] coord [%c0, %c0] : memref<256xf8E5M2, #hivm.address_space<ub>> -> !s_e5m2_to_f32
    %d = tla.tensor_desc %dm shape [%c1, %cd, %c1, %c1] stride [%cd, %c1, %c1, %c1] origin_shape [%c1, %cd] coord [%c0, %c0] : memref<64xf32, #hivm.address_space<ub>> -> !d_e5m2_to_f32
    "tla.vector"() ({
      "tla.vec.func"() ({
        %ss = "tla.make_shape"() : () -> !tla.shape<256>
        %ds = "tla.make_shape"() : () -> !tla.shape<64>
        %co = "tla.make_coord"() : () -> !tla.coord<0>
        %st = "tla.tile_view"(%s, %ss, %co) : (!s_e5m2_to_f32, !tla.shape<256>, !tla.coord<0>) -> !s_e5m2_to_f32
        %dt = "tla.tile_view"(%d, %ds, %co) : (!d_e5m2_to_f32, !tla.shape<64>, !tla.coord<0>) -> !d_e5m2_to_f32
        %r = "tla.load"(%st) : (!s_e5m2_to_f32) -> !tla.vector<256xf8E5M2>
        // trait = [reg_slot, sat_mode, round_mode]
        %o = "tla.cast"(%r) <{trait = array<i32: 0, 0, 4>}> : (!tla.vector<256xf8E5M2>) -> !tla.vector<64xf32>
        "tla.store"(%dt, %o) <{operandSegmentSizes = array<i32: 1, 1, 0, 0, 0>}> : (!d_e5m2_to_f32, !tla.vector<64xf32>) -> ()
      }) {mode = "simd"} : () -> ()
    }) : () -> ()
    return
  }

// CHECK-LABEL: func.func @f32_to_e5m2
// CHECK: "hivm_regbaseintrins.intr.hivm.vcvtff.f322f8e5m2.x"
// CHECK-NOT: ave.hir.vtruncf
  func.func @f32_to_e5m2(
      %sm: memref<64xf32, #hivm.address_space<ub>>,
      %dm: memref<256xf8E5M2, #hivm.address_space<ub>>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %cs = arith.constant 64 : index
    %cd = arith.constant 256 : index
    %s = tla.tensor_desc %sm shape [%c1, %cs, %c1, %c1] stride [%cs, %c1, %c1, %c1] origin_shape [%c1, %cs] coord [%c0, %c0] : memref<64xf32, #hivm.address_space<ub>> -> !s_f32_to_e5m2
    %d = tla.tensor_desc %dm shape [%c1, %cd, %c1, %c1] stride [%cd, %c1, %c1, %c1] origin_shape [%c1, %cd] coord [%c0, %c0] : memref<256xf8E5M2, #hivm.address_space<ub>> -> !d_f32_to_e5m2
    "tla.vector"() ({
      "tla.vec.func"() ({
        %ss = "tla.make_shape"() : () -> !tla.shape<64>
        %ds = "tla.make_shape"() : () -> !tla.shape<256>
        %co = "tla.make_coord"() : () -> !tla.coord<0>
        %st = "tla.tile_view"(%s, %ss, %co) : (!s_f32_to_e5m2, !tla.shape<64>, !tla.coord<0>) -> !s_f32_to_e5m2
        %dt = "tla.tile_view"(%d, %ds, %co) : (!d_f32_to_e5m2, !tla.shape<256>, !tla.coord<0>) -> !d_f32_to_e5m2
        %r = "tla.load"(%st) : (!s_f32_to_e5m2) -> !tla.vector<64xf32>
        // trait = [reg_slot, sat_mode, round_mode]
        %o = "tla.cast"(%r) <{trait = array<i32: 0, 2, 4>}> : (!tla.vector<64xf32>) -> !tla.vector<256xf8E5M2>
        "tla.store"(%dt, %o) <{operandSegmentSizes = array<i32: 1, 1, 0, 0, 0>}> : (!d_f32_to_e5m2, !tla.vector<256xf8E5M2>) -> ()
      }) {mode = "simd"} : () -> ()
    }) : () -> ()
    return
  }

// E5M2 and f16 share their sign/exponent layout, so this exact widening path
// is an integer 2x widen followed by a left shift.
// CHECK-LABEL: func.func @e5m2_to_f16
// CHECK: %[[ALL_B8:.*]] = ave.hir.pge <ALL> : vector<256xi1>
// CHECK: %[[M3:.*]] = ave.hir.pge <M3> : vector<256xi1>
// CHECK: %[[SRC_BITS:.*]] = vector.bitcast %{{.*}} : vector<256xf8E5M2> to vector<256xi8>
// CHECK: %[[MASKED_BITS:.*]] = ave.hir.vand %[[SRC_BITS]], %[[SRC_BITS]], %[[M3]] : vector<256xi8>, vector<256xi1>
// CHECK: %[[WIDENED:.*]] = ave.hir.vextsi %[[MASKED_BITS]], %[[ALL_B8]] {{.*}}<part_even>{{.*}} : vector<256xi8>, vector<128xi16>
// CHECK: %[[SHIFT:.*]] = arith.constant 8 : i16
// CHECK: %[[SHIFTED:.*]] = ave.hir.vshls %[[WIDENED]], %[[SHIFT]], {{.*}} : vector<128xi16>
// CHECK: vector.bitcast %[[SHIFTED]] : vector<128xi16> to vector<128xf16>
// CHECK-NOT: ave.hir.vextf
  func.func @e5m2_to_f16(
      %sm: memref<256xf8E5M2, #hivm.address_space<ub>>,
      %dm: memref<128xf16, #hivm.address_space<ub>>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %cs = arith.constant 256 : index
    %cd = arith.constant 128 : index
    %s = tla.tensor_desc %sm shape [%c1, %cs, %c1, %c1] stride [%cs, %c1, %c1, %c1] origin_shape [%c1, %cs] coord [%c0, %c0] : memref<256xf8E5M2, #hivm.address_space<ub>> -> !s_e5m2_to_f16
    %d = tla.tensor_desc %dm shape [%c1, %cd, %c1, %c1] stride [%cd, %c1, %c1, %c1] origin_shape [%c1, %cd] coord [%c0, %c0] : memref<128xf16, #hivm.address_space<ub>> -> !d_e5m2_to_f16
    "tla.vector"() ({
      "tla.vec.func"() ({
        %ss = "tla.make_shape"() : () -> !tla.shape<256>
        %ds = "tla.make_shape"() : () -> !tla.shape<128>
        %co = "tla.make_coord"() : () -> !tla.coord<0>
        %st = "tla.tile_view"(%s, %ss, %co) : (!s_e5m2_to_f16, !tla.shape<256>, !tla.coord<0>) -> !s_e5m2_to_f16
        %dt = "tla.tile_view"(%d, %ds, %co) : (!d_e5m2_to_f16, !tla.shape<128>, !tla.coord<0>) -> !d_e5m2_to_f16
        %r = "tla.load"(%st) : (!s_e5m2_to_f16) -> !tla.vector<256xf8E5M2>
        // trait = [reg_slot, sat_mode, round_mode]
        %mask = "tla.create_mask"() {pattern = "M3", dtype = f8E5M2} : () -> !tla.mask<256>
        %o = "tla.cast"(%r, %mask) <{trait = array<i32: 0, 0, 4>}> : (!tla.vector<256xf8E5M2>, !tla.mask<256>) -> !tla.vector<128xf16>
        "tla.store"(%dt, %o) <{operandSegmentSizes = array<i32: 1, 1, 0, 0, 0>}> : (!d_e5m2_to_f16, !tla.vector<128xf16>) -> ()
      }) {mode = "simd"} : () -> ()
    }) : () -> ()
    return
  }

// CHECK-LABEL: func.func @e4m3_to_f16
// CHECK: %[[ALL_B8:.*]] = ave.hir.pge <ALL> : vector<256xi1>
// CHECK: %[[M3:.*]] = ave.hir.pge <M3> : vector<256xi1>
// CHECK: %[[SRC_BITS:.*]] = vector.bitcast %{{.*}} : vector<256xf8E4M3FN> to vector<256xi8>
// CHECK: %[[MASKED_BITS:.*]] = ave.hir.vand %[[SRC_BITS]], %[[SRC_BITS]], %[[M3]] : vector<256xi8>, vector<256xi1>
// CHECK: %[[MASKED_FP8:.*]] = vector.bitcast %[[MASKED_BITS]] : vector<256xi8> to vector<256xf8E4M3FN>
// CHECK: %[[PP0:.*]] = arith.constant 0 : i32
// CHECK-NEXT: %[[F32_EVEN:.*]] = "hivm_regbaseintrins.intr.hivm.vcvtff.f8e4m32f32.x"(%[[MASKED_FP8]], %[[ALL_B8]], %[[PP0]]) : (vector<256xf8E4M3FN>, vector<256xi1>, i32) -> vector<64xf32>
// CHECK: %[[PP2:.*]] = arith.constant 2 : i32
// CHECK-NEXT: %[[F32_ODD:.*]] = "hivm_regbaseintrins.intr.hivm.vcvtff.f8e4m32f32.x"(%[[MASKED_FP8]], %[[ALL_B8]], %[[PP2]]) : (vector<256xf8E4M3FN>, vector<256xi1>, i32) -> vector<64xf32>
// CHECK: ave.hir.vtruncf %[[F32_EVEN]], <rint>, false, <part_even>, {{.*}} : vector<64xf32>, vector<128xf16>
// CHECK: ave.hir.vtruncf %[[F32_ODD]], <rint>, false, <part_odd>, {{.*}} : vector<64xf32>, vector<128xf16>
// CHECK: ave.hir.vor {{.*}} : vector<128xi16>
// CHECK-NOT: ave.hir.vextf
  func.func @e4m3_to_f16(
      %sm: memref<256xf8E4M3FN, #hivm.address_space<ub>>,
      %dm: memref<128xf16, #hivm.address_space<ub>>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %cs = arith.constant 256 : index
    %cd = arith.constant 128 : index
    %s = tla.tensor_desc %sm shape [%c1, %cs, %c1, %c1] stride [%cs, %c1, %c1, %c1] origin_shape [%c1, %cs] coord [%c0, %c0] : memref<256xf8E4M3FN, #hivm.address_space<ub>> -> !s_e4m3_to_f16
    %d = tla.tensor_desc %dm shape [%c1, %cd, %c1, %c1] stride [%cd, %c1, %c1, %c1] origin_shape [%c1, %cd] coord [%c0, %c0] : memref<128xf16, #hivm.address_space<ub>> -> !d_e4m3_to_f16
    "tla.vector"() ({
      "tla.vec.func"() ({
        %ss = "tla.make_shape"() : () -> !tla.shape<256>
        %ds = "tla.make_shape"() : () -> !tla.shape<128>
        %co = "tla.make_coord"() : () -> !tla.coord<0>
        %st = "tla.tile_view"(%s, %ss, %co) : (!s_e4m3_to_f16, !tla.shape<256>, !tla.coord<0>) -> !s_e4m3_to_f16
        %dt = "tla.tile_view"(%d, %ds, %co) : (!d_e4m3_to_f16, !tla.shape<128>, !tla.coord<0>) -> !d_e4m3_to_f16
        %r = "tla.load"(%st) : (!s_e4m3_to_f16) -> !tla.vector<256xf8E4M3FN>
        // trait = [reg_slot, sat_mode, round_mode]
        %mask = "tla.create_mask"() {pattern = "M3", dtype = f8E4M3FN} : () -> !tla.mask<256>
        %o = "tla.cast"(%r, %mask) <{trait = array<i32: 0, 0, 4>}> : (!tla.vector<256xf8E4M3FN>, !tla.mask<256>) -> !tla.vector<128xf16>
        "tla.store"(%dt, %o) <{operandSegmentSizes = array<i32: 1, 1, 0, 0, 0>}> : (!d_e4m3_to_f16, !tla.vector<128xf16>) -> ()
      }) {mode = "simd"} : () -> ()
    }) : () -> ()
    return
  }

// CHECK-LABEL: func.func @f16_to_e4m3
// CHECK: ave.hir.vand
// CHECK: ave.hir.vextf {{.*}}<part_even>{{.*}} : vector<128xf16>, vector<64xf32>
// CHECK: ave.hir.vextf {{.*}}<part_odd>{{.*}} : vector<128xf16>, vector<64xf32>
// CHECK: %[[RINT:.*]] = arith.constant 0 : i32
// CHECK-NEXT: %[[NOSAT:.*]] = arith.constant 0 : i32
// CHECK-NEXT: %[[PP1:.*]] = arith.constant 1 : i32
// CHECK-NEXT: "hivm_regbaseintrins.intr.hivm.vcvtff.f322f8e4m3.x"(%{{.*}}, %{{.*}}, %[[RINT]], %[[NOSAT]], %[[PP1]]) : (vector<64xf32>, vector<256xi1>, i32, i32, i32) -> vector<256xf8E4M3FN>
// CHECK: %[[PP3:.*]] = arith.constant 3 : i32
// CHECK-NEXT: "hivm_regbaseintrins.intr.hivm.vcvtff.f322f8e4m3.x"(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %[[PP3]]) : (vector<64xf32>, vector<256xi1>, i32, i32, i32) -> vector<256xf8E4M3FN>
// CHECK: ave.hir.vor
// CHECK-NOT: ave.hir.vtruncf
  func.func @f16_to_e4m3(
      %sm: memref<128xf16, #hivm.address_space<ub>>,
      %dm: memref<256xf8E4M3FN, #hivm.address_space<ub>>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %cs = arith.constant 128 : index
    %cd = arith.constant 256 : index
    %s = tla.tensor_desc %sm shape [%c1, %cs, %c1, %c1] stride [%cs, %c1, %c1, %c1] origin_shape [%c1, %cs] coord [%c0, %c0] : memref<128xf16, #hivm.address_space<ub>> -> !s_f16_to_e4m3
    %d = tla.tensor_desc %dm shape [%c1, %cd, %c1, %c1] stride [%cd, %c1, %c1, %c1] origin_shape [%c1, %cd] coord [%c0, %c0] : memref<256xf8E4M3FN, #hivm.address_space<ub>> -> !d_f16_to_e4m3
    "tla.vector"() ({
      "tla.vec.func"() ({
        %ss = "tla.make_shape"() : () -> !tla.shape<128>
        %ds = "tla.make_shape"() : () -> !tla.shape<256>
        %co = "tla.make_coord"() : () -> !tla.coord<0>
        %st = "tla.tile_view"(%s, %ss, %co) : (!s_f16_to_e4m3, !tla.shape<128>, !tla.coord<0>) -> !s_f16_to_e4m3
        %dt = "tla.tile_view"(%d, %ds, %co) : (!d_f16_to_e4m3, !tla.shape<256>, !tla.coord<0>) -> !d_f16_to_e4m3
        %r = "tla.load"(%st) : (!s_f16_to_e4m3) -> !tla.vector<128xf16>
        %mask = "tla.create_mask"() {pattern = "M3", dtype = f16} : () -> !tla.mask<128>
        %o = "tla.cast"(%r, %mask) <{trait = array<i32: 1, 2, 4>}> : (!tla.vector<128xf16>, !tla.mask<128>) -> !tla.vector<256xf8E4M3FN>
        "tla.store"(%dt, %o) <{operandSegmentSizes = array<i32: 1, 1, 0, 0, 0>}> : (!d_f16_to_e4m3, !tla.vector<256xf8E4M3FN>) -> ()
      }) {mode = "simd"} : () -> ()
    }) : () -> ()
    return
  }

// CHECK-LABEL: func.func @e4m3_to_bf16
// CHECK: %[[PP1:.*]] = arith.constant 1 : i32
// CHECK-NEXT: %[[F32_EVEN:.*]] = "hivm_regbaseintrins.intr.hivm.vcvtff.f8e4m32f32.x"(%{{.*}}, %{{.*}}, %[[PP1]]) : (vector<256xf8E4M3FN>, vector<256xi1>, i32) -> vector<64xf32>
// CHECK: %[[PP3:.*]] = arith.constant 3 : i32
// CHECK-NEXT: %[[F32_ODD:.*]] = "hivm_regbaseintrins.intr.hivm.vcvtff.f8e4m32f32.x"(%{{.*}}, %{{.*}}, %[[PP3]]) : (vector<256xf8E4M3FN>, vector<256xi1>, i32) -> vector<64xf32>
// CHECK: ave.hir.vtruncf %[[F32_EVEN]], <rint>, false, <part_even>, {{.*}} : vector<64xf32>, vector<128xbf16>
// CHECK: ave.hir.vtruncf %[[F32_ODD]], <rint>, false, <part_odd>, {{.*}} : vector<64xf32>, vector<128xbf16>
// CHECK: ave.hir.vor {{.*}} : vector<128xi16>
// CHECK-NOT: ave.hir.vextf
  func.func @e4m3_to_bf16(
      %sm: memref<256xf8E4M3FN, #hivm.address_space<ub>>,
      %dm: memref<128xbf16, #hivm.address_space<ub>>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %cs = arith.constant 256 : index
    %cd = arith.constant 128 : index
    %s = tla.tensor_desc %sm shape [%c1, %cs, %c1, %c1] stride [%cs, %c1, %c1, %c1] origin_shape [%c1, %cs] coord [%c0, %c0] : memref<256xf8E4M3FN, #hivm.address_space<ub>> -> !s_e4m3_to_bf16
    %d = tla.tensor_desc %dm shape [%c1, %cd, %c1, %c1] stride [%cd, %c1, %c1, %c1] origin_shape [%c1, %cd] coord [%c0, %c0] : memref<128xbf16, #hivm.address_space<ub>> -> !d_e4m3_to_bf16
    "tla.vector"() ({
      "tla.vec.func"() ({
        %ss = "tla.make_shape"() : () -> !tla.shape<256>
        %ds = "tla.make_shape"() : () -> !tla.shape<128>
        %co = "tla.make_coord"() : () -> !tla.coord<0>
        %st = "tla.tile_view"(%s, %ss, %co) : (!s_e4m3_to_bf16, !tla.shape<256>, !tla.coord<0>) -> !s_e4m3_to_bf16
        %dt = "tla.tile_view"(%d, %ds, %co) : (!d_e4m3_to_bf16, !tla.shape<128>, !tla.coord<0>) -> !d_e4m3_to_bf16
        %r = "tla.load"(%st) : (!s_e4m3_to_bf16) -> !tla.vector<256xf8E4M3FN>
        // trait = [reg_slot, sat_mode, round_mode]
        %o = "tla.cast"(%r) <{trait = array<i32: 1, 0, 4>}> : (!tla.vector<256xf8E4M3FN>) -> !tla.vector<128xbf16>
        "tla.store"(%dt, %o) <{operandSegmentSizes = array<i32: 1, 1, 0, 0, 0>}> : (!d_e4m3_to_bf16, !tla.vector<128xbf16>) -> ()
      }) {mode = "simd"} : () -> ()
    }) : () -> ()
    return
  }

// CHECK-LABEL: func.func @bf16_to_e4m3
// CHECK: %[[F32_EVEN:.*]] = ave.hir.vextf {{.*}}<part_even>{{.*}} : vector<128xbf16>, vector<64xf32>
// CHECK: %[[F32_ODD:.*]] = ave.hir.vextf {{.*}}<part_odd>{{.*}} : vector<128xbf16>, vector<64xf32>
// CHECK: %[[RINT:.*]] = arith.constant 0 : i32
// CHECK-NEXT: %[[NOSAT:.*]] = arith.constant 0 : i32
// CHECK-NEXT: %[[PP0:.*]] = arith.constant 0 : i32
// CHECK-NEXT: "hivm_regbaseintrins.intr.hivm.vcvtff.f322f8e4m3.x"(%[[F32_EVEN]], %{{.*}}, %[[RINT]], %[[NOSAT]], %[[PP0]]) : (vector<64xf32>, vector<256xi1>, i32, i32, i32) -> vector<256xf8E4M3FN>
// CHECK: %[[PP2:.*]] = arith.constant 2 : i32
// CHECK-NEXT: "hivm_regbaseintrins.intr.hivm.vcvtff.f322f8e4m3.x"(%[[F32_ODD]], %{{.*}}, %{{.*}}, %{{.*}}, %[[PP2]]) : (vector<64xf32>, vector<256xi1>, i32, i32, i32) -> vector<256xf8E4M3FN>
// CHECK: ave.hir.vor {{.*}} : vector<256xi8>
// CHECK-NOT: ave.hir.vtruncf
  func.func @bf16_to_e4m3(
      %sm: memref<128xbf16, #hivm.address_space<ub>>,
      %dm: memref<256xf8E4M3FN, #hivm.address_space<ub>>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %cs = arith.constant 128 : index
    %cd = arith.constant 256 : index
    %s = tla.tensor_desc %sm shape [%c1, %cs, %c1, %c1] stride [%cs, %c1, %c1, %c1] origin_shape [%c1, %cs] coord [%c0, %c0] : memref<128xbf16, #hivm.address_space<ub>> -> !s_bf16_to_e4m3
    %d = tla.tensor_desc %dm shape [%c1, %cd, %c1, %c1] stride [%cd, %c1, %c1, %c1] origin_shape [%c1, %cd] coord [%c0, %c0] : memref<256xf8E4M3FN, #hivm.address_space<ub>> -> !d_bf16_to_e4m3
    "tla.vector"() ({
      "tla.vec.func"() ({
        %ss = "tla.make_shape"() : () -> !tla.shape<128>
        %ds = "tla.make_shape"() : () -> !tla.shape<256>
        %co = "tla.make_coord"() : () -> !tla.coord<0>
        %st = "tla.tile_view"(%s, %ss, %co) : (!s_bf16_to_e4m3, !tla.shape<128>, !tla.coord<0>) -> !s_bf16_to_e4m3
        %dt = "tla.tile_view"(%d, %ds, %co) : (!d_bf16_to_e4m3, !tla.shape<256>, !tla.coord<0>) -> !d_bf16_to_e4m3
        %r = "tla.load"(%st) : (!s_bf16_to_e4m3) -> !tla.vector<128xbf16>
        // trait = [reg_slot, sat_mode, round_mode]
        %o = "tla.cast"(%r) <{trait = array<i32: 0, 2, 5>}> : (!tla.vector<128xbf16>) -> !tla.vector<256xf8E4M3FN>
        "tla.store"(%dt, %o) <{operandSegmentSizes = array<i32: 1, 1, 0, 0, 0>}> : (!d_bf16_to_e4m3, !tla.vector<256xf8E4M3FN>) -> ()
      }) {mode = "simd"} : () -> ()
    }) : () -> ()
    return
  }

// CHECK-LABEL: func.func @f16_to_e5m2
// CHECK: %[[F32_EVEN:.*]] = ave.hir.vextf {{.*}}<part_even>{{.*}} : vector<128xf16>, vector<64xf32>
// CHECK: %[[F32_ODD:.*]] = ave.hir.vextf {{.*}}<part_odd>{{.*}} : vector<128xf16>, vector<64xf32>
// CHECK: %[[RINT_EVEN:.*]] = arith.constant 0 : i32
// CHECK-NEXT: %[[SAT_EVEN:.*]] = arith.constant 1 : i32
// CHECK-NEXT: %[[PP0:.*]] = arith.constant 0 : i32
// CHECK-NEXT: "hivm_regbaseintrins.intr.hivm.vcvtff.f322f8e5m2.x"(%[[F32_EVEN]], %{{.*}}, %[[RINT_EVEN]], %[[SAT_EVEN]], %[[PP0]]) : (vector<64xf32>, vector<256xi1>, i32, i32, i32) -> vector<256xf8E5M2>
// CHECK: %[[RINT_ODD:.*]] = arith.constant 0 : i32
// CHECK-NEXT: %[[SAT_ODD:.*]] = arith.constant 1 : i32
// CHECK-NEXT: %[[PP2:.*]] = arith.constant 2 : i32
// CHECK-NEXT: "hivm_regbaseintrins.intr.hivm.vcvtff.f322f8e5m2.x"(%[[F32_ODD]], %{{.*}}, %[[RINT_ODD]], %[[SAT_ODD]], %[[PP2]]) : (vector<64xf32>, vector<256xi1>, i32, i32, i32) -> vector<256xf8E5M2>
// CHECK: ave.hir.vor {{.*}} : vector<256xi8>
// CHECK-NOT: ave.hir.vtruncf
  func.func @f16_to_e5m2(
      %sm: memref<128xf16, #hivm.address_space<ub>>,
      %dm: memref<256xf8E5M2, #hivm.address_space<ub>>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %cs = arith.constant 128 : index
    %cd = arith.constant 256 : index
    %s = tla.tensor_desc %sm shape [%c1, %cs, %c1, %c1] stride [%cs, %c1, %c1, %c1] origin_shape [%c1, %cs] coord [%c0, %c0] : memref<128xf16, #hivm.address_space<ub>> -> !s_f16_to_e5m2
    %d = tla.tensor_desc %dm shape [%c1, %cd, %c1, %c1] stride [%cd, %c1, %c1, %c1] origin_shape [%c1, %cd] coord [%c0, %c0] : memref<256xf8E5M2, #hivm.address_space<ub>> -> !d_f16_to_e5m2
    "tla.vector"() ({
      "tla.vec.func"() ({
        %ss = "tla.make_shape"() : () -> !tla.shape<128>
        %ds = "tla.make_shape"() : () -> !tla.shape<256>
        %co = "tla.make_coord"() : () -> !tla.coord<0>
        %st = "tla.tile_view"(%s, %ss, %co) : (!s_f16_to_e5m2, !tla.shape<128>, !tla.coord<0>) -> !s_f16_to_e5m2
        %dt = "tla.tile_view"(%d, %ds, %co) : (!d_f16_to_e5m2, !tla.shape<256>, !tla.coord<0>) -> !d_f16_to_e5m2
        %r = "tla.load"(%st) : (!s_f16_to_e5m2) -> !tla.vector<128xf16>
        // trait = [reg_slot, sat_mode, round_mode]
        %o = "tla.cast"(%r) <{trait = array<i32: 0, 1, 4>}> : (!tla.vector<128xf16>) -> !tla.vector<256xf8E5M2>
        "tla.store"(%dt, %o) <{operandSegmentSizes = array<i32: 1, 1, 0, 0, 0>}> : (!d_f16_to_e5m2, !tla.vector<256xf8E5M2>) -> ()
      }) {mode = "simd"} : () -> ()
    }) : () -> ()
    return
  }

// CHECK-LABEL: func.func @e5m2_to_bf16
// CHECK: %[[PP0:.*]] = arith.constant 0 : i32
// CHECK-NEXT: %[[F32_EVEN:.*]] = "hivm_regbaseintrins.intr.hivm.vcvtff.f8e5m22f32.x"(%{{.*}}, %{{.*}}, %[[PP0]]) : (vector<256xf8E5M2>, vector<256xi1>, i32) -> vector<64xf32>
// CHECK: %[[PP2:.*]] = arith.constant 2 : i32
// CHECK-NEXT: %[[F32_ODD:.*]] = "hivm_regbaseintrins.intr.hivm.vcvtff.f8e5m22f32.x"(%{{.*}}, %{{.*}}, %[[PP2]]) : (vector<256xf8E5M2>, vector<256xi1>, i32) -> vector<64xf32>
// CHECK: ave.hir.vtruncf %[[F32_EVEN]], <rint>, false, <part_even>, {{.*}} : vector<64xf32>, vector<128xbf16>
// CHECK: ave.hir.vtruncf %[[F32_ODD]], <rint>, false, <part_odd>, {{.*}} : vector<64xf32>, vector<128xbf16>
// CHECK: ave.hir.vor {{.*}} : vector<128xi16>
// CHECK-NOT: ave.hir.vextf
  func.func @e5m2_to_bf16(
      %sm: memref<256xf8E5M2, #hivm.address_space<ub>>,
      %dm: memref<128xbf16, #hivm.address_space<ub>>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %cs = arith.constant 256 : index
    %cd = arith.constant 128 : index
    %s = tla.tensor_desc %sm shape [%c1, %cs, %c1, %c1] stride [%cs, %c1, %c1, %c1] origin_shape [%c1, %cs] coord [%c0, %c0] : memref<256xf8E5M2, #hivm.address_space<ub>> -> !s_e5m2_to_bf16
    %d = tla.tensor_desc %dm shape [%c1, %cd, %c1, %c1] stride [%cd, %c1, %c1, %c1] origin_shape [%c1, %cd] coord [%c0, %c0] : memref<128xbf16, #hivm.address_space<ub>> -> !d_e5m2_to_bf16
    "tla.vector"() ({
      "tla.vec.func"() ({
        %ss = "tla.make_shape"() : () -> !tla.shape<256>
        %ds = "tla.make_shape"() : () -> !tla.shape<128>
        %co = "tla.make_coord"() : () -> !tla.coord<0>
        %st = "tla.tile_view"(%s, %ss, %co) : (!s_e5m2_to_bf16, !tla.shape<256>, !tla.coord<0>) -> !s_e5m2_to_bf16
        %dt = "tla.tile_view"(%d, %ds, %co) : (!d_e5m2_to_bf16, !tla.shape<128>, !tla.coord<0>) -> !d_e5m2_to_bf16
        %r = "tla.load"(%st) : (!s_e5m2_to_bf16) -> !tla.vector<256xf8E5M2>
        // trait = [reg_slot, sat_mode, round_mode]
        %o = "tla.cast"(%r) <{trait = array<i32: 0, 0, 4>}> : (!tla.vector<256xf8E5M2>) -> !tla.vector<128xbf16>
        "tla.store"(%dt, %o) <{operandSegmentSizes = array<i32: 1, 1, 0, 0, 0>}> : (!d_e5m2_to_bf16, !tla.vector<128xbf16>) -> ()
      }) {mode = "simd"} : () -> ()
    }) : () -> ()
    return
  }

// CHECK-LABEL: func.func @bf16_to_e5m2
// CHECK: %[[F32_EVEN:.*]] = ave.hir.vextf {{.*}}<part_even>{{.*}} : vector<128xbf16>, vector<64xf32>
// CHECK: %[[F32_ODD:.*]] = ave.hir.vextf {{.*}}<part_odd>{{.*}} : vector<128xbf16>, vector<64xf32>
// CHECK: %[[PP1:.*]] = arith.constant 1 : i32
// CHECK-NEXT: "hivm_regbaseintrins.intr.hivm.vcvtff.f322f8e5m2.x"(%[[F32_EVEN]], %{{.*}}, %{{.*}}, %{{.*}}, %[[PP1]]) : (vector<64xf32>, vector<256xi1>, i32, i32, i32) -> vector<256xf8E5M2>
// CHECK: %[[PP3:.*]] = arith.constant 3 : i32
// CHECK-NEXT: "hivm_regbaseintrins.intr.hivm.vcvtff.f322f8e5m2.x"(%[[F32_ODD]], %{{.*}}, %{{.*}}, %{{.*}}, %[[PP3]]) : (vector<64xf32>, vector<256xi1>, i32, i32, i32) -> vector<256xf8E5M2>
// CHECK: ave.hir.vor {{.*}} : vector<256xi8>
// CHECK-NOT: ave.hir.vtruncf
  func.func @bf16_to_e5m2(
      %sm: memref<128xbf16, #hivm.address_space<ub>>,
      %dm: memref<256xf8E5M2, #hivm.address_space<ub>>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %cs = arith.constant 128 : index
    %cd = arith.constant 256 : index
    %s = tla.tensor_desc %sm shape [%c1, %cs, %c1, %c1] stride [%cs, %c1, %c1, %c1] origin_shape [%c1, %cs] coord [%c0, %c0] : memref<128xbf16, #hivm.address_space<ub>> -> !s_bf16_to_e5m2
    %d = tla.tensor_desc %dm shape [%c1, %cd, %c1, %c1] stride [%cd, %c1, %c1, %c1] origin_shape [%c1, %cd] coord [%c0, %c0] : memref<256xf8E5M2, #hivm.address_space<ub>> -> !d_bf16_to_e5m2
    "tla.vector"() ({
      "tla.vec.func"() ({
        %ss = "tla.make_shape"() : () -> !tla.shape<128>
        %ds = "tla.make_shape"() : () -> !tla.shape<256>
        %co = "tla.make_coord"() : () -> !tla.coord<0>
        %st = "tla.tile_view"(%s, %ss, %co) : (!s_bf16_to_e5m2, !tla.shape<128>, !tla.coord<0>) -> !s_bf16_to_e5m2
        %dt = "tla.tile_view"(%d, %ds, %co) : (!d_bf16_to_e5m2, !tla.shape<256>, !tla.coord<0>) -> !d_bf16_to_e5m2
        %r = "tla.load"(%st) : (!s_bf16_to_e5m2) -> !tla.vector<128xbf16>
        // trait = [reg_slot, sat_mode, round_mode]
        %o = "tla.cast"(%r) <{trait = array<i32: 1, 2, 4>}> : (!tla.vector<128xbf16>) -> !tla.vector<256xf8E5M2>
        "tla.store"(%dt, %o) <{operandSegmentSizes = array<i32: 1, 1, 0, 0, 0>}> : (!d_bf16_to_e5m2, !tla.vector<256xf8E5M2>) -> ()
      }) {mode = "simd"} : () -> ()
    }) : () -> ()
    return
  }
}
