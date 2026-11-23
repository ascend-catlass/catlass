// RUN: %tla_compile %s -o - --mlir-print-ir-after=tla-vector-region-simd 2>&1 | %filecheck %s

// Every fp8 cast leg the vector unit accepts, and how each one lowers.
//
// The ISA carries exactly four fp8 conversions -- VcvtffF322F8E4M3,
// VcvtffF322F8E5M2 and their F32-result inverses -- so only the f32 pairs are a
// single instruction. f16 and bf16 have no fp8 instruction at all and are
// expanded by TlaVectorRegionPass into an even/odd pair through f32, then a
// bitwise OR which reconstructs the complete packed result.
//
// The vector types in each CHECK pin the intermediate, so a lowering that
// emitted one instruction straight between fp8 and f16/bf16 -- the shape of the
// bug this gate exists to catch, where a narrow silently used the f32-source
// instruction against f16 lanes -- fails here even if the op names look right.
//
// Every fp8 leg -- direct or composed -- uses the raw PP-capable regbase
// instructions, because the f8 <-> f32 conversion is 4x wide and its selector
// is the pack pattern pp0..pp3, not the 2x even/odd part the high-level AVE
// float ops carry. HIVMAVEToAVEIntrin forwards `part` verbatim as the pp
// immediate, so building this leg from ave.hir.vextf / ave.hir.vtruncf would
// silently pin pp to 0 and 1 and touch only two of the four byte positions.
// The CHECKs below therefore pin the pp immediates to 0 and 2, which is what
// makes the composed cast select whole elements (dst[j] = src[2j + slot])
// rather than a strided half of them. The f16/bf16 side of each composition
// is a genuine 2x step and stays on the even/odd AVE ops.

!s_e4m3_to_f32 = !tla.tensor<!tla.layout<!tla.shape<256>, !tla.stride<1>, !tla.shape<256>, RowMajor>, !tla.coord<0>, !tla.ptr<f8E4M3FN, ub, 1>>
!d_e4m3_to_f32 = !tla.tensor<!tla.layout<!tla.shape<64>, !tla.stride<1>, !tla.shape<64>, RowMajor>, !tla.coord<0>, !tla.ptr<f32, ub, 4>>
!s_f32_to_e4m3 = !tla.tensor<!tla.layout<!tla.shape<64>, !tla.stride<1>, !tla.shape<64>, RowMajor>, !tla.coord<0>, !tla.ptr<f32, ub, 4>>
!d_f32_to_e4m3 = !tla.tensor<!tla.layout<!tla.shape<256>, !tla.stride<1>, !tla.shape<256>, RowMajor>, !tla.coord<0>, !tla.ptr<f8E4M3FN, ub, 1>>
!s_e4m3_to_f16 = !tla.tensor<!tla.layout<!tla.shape<256>, !tla.stride<1>, !tla.shape<256>, RowMajor>, !tla.coord<0>, !tla.ptr<f8E4M3FN, ub, 1>>
!d_e4m3_to_f16 = !tla.tensor<!tla.layout<!tla.shape<128>, !tla.stride<1>, !tla.shape<128>, RowMajor>, !tla.coord<0>, !tla.ptr<f16, ub, 2>>
!s_f16_to_e4m3 = !tla.tensor<!tla.layout<!tla.shape<128>, !tla.stride<1>, !tla.shape<128>, RowMajor>, !tla.coord<0>, !tla.ptr<f16, ub, 2>>
!d_f16_to_e4m3 = !tla.tensor<!tla.layout<!tla.shape<256>, !tla.stride<1>, !tla.shape<256>, RowMajor>, !tla.coord<0>, !tla.ptr<f8E4M3FN, ub, 1>>
!s_e4m3_to_bf16 = !tla.tensor<!tla.layout<!tla.shape<256>, !tla.stride<1>, !tla.shape<256>, RowMajor>, !tla.coord<0>, !tla.ptr<f8E4M3FN, ub, 1>>
!d_e4m3_to_bf16 = !tla.tensor<!tla.layout<!tla.shape<128>, !tla.stride<1>, !tla.shape<128>, RowMajor>, !tla.coord<0>, !tla.ptr<bf16, ub, 2>>
!s_bf16_to_e4m3 = !tla.tensor<!tla.layout<!tla.shape<128>, !tla.stride<1>, !tla.shape<128>, RowMajor>, !tla.coord<0>, !tla.ptr<bf16, ub, 2>>
!d_bf16_to_e4m3 = !tla.tensor<!tla.layout<!tla.shape<256>, !tla.stride<1>, !tla.shape<256>, RowMajor>, !tla.coord<0>, !tla.ptr<f8E4M3FN, ub, 1>>
!s_e5m2_to_f32 = !tla.tensor<!tla.layout<!tla.shape<256>, !tla.stride<1>, !tla.shape<256>, RowMajor>, !tla.coord<0>, !tla.ptr<f8E5M2, ub, 1>>
!d_e5m2_to_f32 = !tla.tensor<!tla.layout<!tla.shape<64>, !tla.stride<1>, !tla.shape<64>, RowMajor>, !tla.coord<0>, !tla.ptr<f32, ub, 4>>
!s_f32_to_e5m2 = !tla.tensor<!tla.layout<!tla.shape<64>, !tla.stride<1>, !tla.shape<64>, RowMajor>, !tla.coord<0>, !tla.ptr<f32, ub, 4>>
!d_f32_to_e5m2 = !tla.tensor<!tla.layout<!tla.shape<256>, !tla.stride<1>, !tla.shape<256>, RowMajor>, !tla.coord<0>, !tla.ptr<f8E5M2, ub, 1>>
!s_e5m2_to_f16 = !tla.tensor<!tla.layout<!tla.shape<256>, !tla.stride<1>, !tla.shape<256>, RowMajor>, !tla.coord<0>, !tla.ptr<f8E5M2, ub, 1>>
!d_e5m2_to_f16 = !tla.tensor<!tla.layout<!tla.shape<128>, !tla.stride<1>, !tla.shape<128>, RowMajor>, !tla.coord<0>, !tla.ptr<f16, ub, 2>>
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
        %o = "tla.cast"(%r) <{trait = array<i32: 0, 0, 0>}> : (!tla.vector<256xf8E4M3FN>) -> !tla.vector<64xf32>
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
        %o = "tla.cast"(%r) <{trait = array<i32: 0, 0, 0>}> : (!tla.vector<64xf32>) -> !tla.vector<256xf8E4M3FN>
        "tla.store"(%dt, %o) <{operandSegmentSizes = array<i32: 1, 1, 0, 0, 0>}> : (!d_f32_to_e4m3, !tla.vector<256xf8E4M3FN>) -> ()
      }) {mode = "simd"} : () -> ()
    }) : () -> ()
    return
  }

// CHECK-LABEL: func.func @e4m3_to_f16
// CHECK: %[[PP0:.*]] = arith.constant 0 : i32
// CHECK-NEXT: "hivm_regbaseintrins.intr.hivm.vcvtff.f8e4m32f32.x"(%{{.*}}, %{{.*}}, %[[PP0]]) : (vector<256xf8E4M3FN>, vector<256xi1>, i32) -> vector<64xf32>
// CHECK: %[[PP2:.*]] = arith.constant 2 : i32
// CHECK-NEXT: "hivm_regbaseintrins.intr.hivm.vcvtff.f8e4m32f32.x"(%{{.*}}, %{{.*}}, %[[PP2]]) : (vector<256xf8E4M3FN>, vector<256xi1>, i32) -> vector<64xf32>
// CHECK: ave.hir.vtruncf {{.*}}<part_even>{{.*}} : vector<64xf32>, vector<128xf16>
// CHECK: ave.hir.vtruncf {{.*}}<part_odd>{{.*}} : vector<64xf32>, vector<128xf16>
// CHECK: ave.hir.vor
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
        %o = "tla.cast"(%r) <{trait = array<i32: 0, 0, 0>}> : (!tla.vector<256xf8E4M3FN>) -> !tla.vector<128xf16>
        "tla.store"(%dt, %o) <{operandSegmentSizes = array<i32: 1, 1, 0, 0, 0>}> : (!d_e4m3_to_f16, !tla.vector<128xf16>) -> ()
      }) {mode = "simd"} : () -> ()
    }) : () -> ()
    return
  }

// CHECK-LABEL: func.func @f16_to_e4m3
// CHECK: ave.hir.vextf {{.*}}<part_even>{{.*}} : vector<128xf16>, vector<64xf32>
// CHECK: ave.hir.vextf {{.*}}<part_odd>{{.*}} : vector<128xf16>, vector<64xf32>
// CHECK: arith.constant 0 : i32
// CHECK-NEXT: %[[PP0:.*]] = arith.constant 0 : i32
// CHECK-NEXT: "hivm_regbaseintrins.intr.hivm.vcvtff.f322f8e4m3.x"(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %[[PP0]]) : (vector<64xf32>, vector<256xi1>, i32, i32, i32) -> vector<256xf8E4M3FN>
// CHECK: %[[PP2:.*]] = arith.constant 2 : i32
// CHECK-NEXT: "hivm_regbaseintrins.intr.hivm.vcvtff.f322f8e4m3.x"(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %[[PP2]]) : (vector<64xf32>, vector<256xi1>, i32, i32, i32) -> vector<256xf8E4M3FN>
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
        // trait = [reg_slot, sat_mode, round_mode]
        %o = "tla.cast"(%r) <{trait = array<i32: 0, 0, 0>}> : (!tla.vector<128xf16>) -> !tla.vector<256xf8E4M3FN>
        "tla.store"(%dt, %o) <{operandSegmentSizes = array<i32: 1, 1, 0, 0, 0>}> : (!d_f16_to_e4m3, !tla.vector<256xf8E4M3FN>) -> ()
      }) {mode = "simd"} : () -> ()
    }) : () -> ()
    return
  }

// CHECK-LABEL: func.func @e4m3_to_bf16
// CHECK: %[[PP0:.*]] = arith.constant 0 : i32
// CHECK-NEXT: "hivm_regbaseintrins.intr.hivm.vcvtff.f8e4m32f32.x"(%{{.*}}, %{{.*}}, %[[PP0]]) : (vector<256xf8E4M3FN>, vector<256xi1>, i32) -> vector<64xf32>
// CHECK: %[[PP2:.*]] = arith.constant 2 : i32
// CHECK-NEXT: "hivm_regbaseintrins.intr.hivm.vcvtff.f8e4m32f32.x"(%{{.*}}, %{{.*}}, %[[PP2]]) : (vector<256xf8E4M3FN>, vector<256xi1>, i32) -> vector<64xf32>
// CHECK: ave.hir.vtruncf {{.*}}<part_even>{{.*}} : vector<64xf32>, vector<128xbf16>
// CHECK: ave.hir.vtruncf {{.*}}<part_odd>{{.*}} : vector<64xf32>, vector<128xbf16>
// CHECK: ave.hir.vor
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
        %o = "tla.cast"(%r) <{trait = array<i32: 0, 0, 0>}> : (!tla.vector<256xf8E4M3FN>) -> !tla.vector<128xbf16>
        "tla.store"(%dt, %o) <{operandSegmentSizes = array<i32: 1, 1, 0, 0, 0>}> : (!d_e4m3_to_bf16, !tla.vector<128xbf16>) -> ()
      }) {mode = "simd"} : () -> ()
    }) : () -> ()
    return
  }

// CHECK-LABEL: func.func @bf16_to_e4m3
// CHECK: ave.hir.vextf {{.*}}<part_even>{{.*}} : vector<128xbf16>, vector<64xf32>
// CHECK: ave.hir.vextf {{.*}}<part_odd>{{.*}} : vector<128xbf16>, vector<64xf32>
// CHECK: arith.constant 0 : i32
// CHECK-NEXT: %[[PP0:.*]] = arith.constant 0 : i32
// CHECK-NEXT: "hivm_regbaseintrins.intr.hivm.vcvtff.f322f8e4m3.x"(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %[[PP0]]) : (vector<64xf32>, vector<256xi1>, i32, i32, i32) -> vector<256xf8E4M3FN>
// CHECK: %[[PP2:.*]] = arith.constant 2 : i32
// CHECK-NEXT: "hivm_regbaseintrins.intr.hivm.vcvtff.f322f8e4m3.x"(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %[[PP2]]) : (vector<64xf32>, vector<256xi1>, i32, i32, i32) -> vector<256xf8E4M3FN>
// CHECK: ave.hir.vor
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
        %o = "tla.cast"(%r) <{trait = array<i32: 0, 0, 0>}> : (!tla.vector<128xbf16>) -> !tla.vector<256xf8E4M3FN>
        "tla.store"(%dt, %o) <{operandSegmentSizes = array<i32: 1, 1, 0, 0, 0>}> : (!d_bf16_to_e4m3, !tla.vector<256xf8E4M3FN>) -> ()
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
        %o = "tla.cast"(%r) <{trait = array<i32: 0, 0, 0>}> : (!tla.vector<256xf8E5M2>) -> !tla.vector<64xf32>
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
        %o = "tla.cast"(%r) <{trait = array<i32: 0, 0, 0>}> : (!tla.vector<64xf32>) -> !tla.vector<256xf8E5M2>
        "tla.store"(%dt, %o) <{operandSegmentSizes = array<i32: 1, 1, 0, 0, 0>}> : (!d_f32_to_e5m2, !tla.vector<256xf8E5M2>) -> ()
      }) {mode = "simd"} : () -> ()
    }) : () -> ()
    return
  }

// E5M2 and f16 share their sign/exponent layout, so this exact widening path
// is an integer 2x widen followed by a left shift. Keep it separate from the
// generic fp8->f32->f16 composition: the latter is correct but takes five
// instructions where this takes two.
// CHECK-LABEL: func.func @e5m2_to_f16
// CHECK: ave.hir.vextsi {{.*}} : vector<256xi8>, vector<128xi16>
// CHECK: ave.hir.vshls {{.*}} : vector<128xi16>
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
        %o = "tla.cast"(%r) <{trait = array<i32: 0, 0, 0>}> : (!tla.vector<256xf8E5M2>) -> !tla.vector<128xf16>
        "tla.store"(%dt, %o) <{operandSegmentSizes = array<i32: 1, 1, 0, 0, 0>}> : (!d_e5m2_to_f16, !tla.vector<128xf16>) -> ()
      }) {mode = "simd"} : () -> ()
    }) : () -> ()
    return
  }

// CHECK-LABEL: func.func @f16_to_e5m2
// CHECK: ave.hir.vextf {{.*}}<part_even>{{.*}} : vector<128xf16>, vector<64xf32>
// CHECK: ave.hir.vextf {{.*}}<part_odd>{{.*}} : vector<128xf16>, vector<64xf32>
// CHECK: arith.constant 0 : i32
// CHECK-NEXT: %[[PP0:.*]] = arith.constant 0 : i32
// CHECK-NEXT: "hivm_regbaseintrins.intr.hivm.vcvtff.f322f8e5m2.x"(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %[[PP0]]) : (vector<64xf32>, vector<256xi1>, i32, i32, i32) -> vector<256xf8E5M2>
// CHECK: %[[PP2:.*]] = arith.constant 2 : i32
// CHECK-NEXT: "hivm_regbaseintrins.intr.hivm.vcvtff.f322f8e5m2.x"(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %[[PP2]]) : (vector<64xf32>, vector<256xi1>, i32, i32, i32) -> vector<256xf8E5M2>
// CHECK: ave.hir.vor
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
        %o = "tla.cast"(%r) <{trait = array<i32: 0, 0, 0>}> : (!tla.vector<128xf16>) -> !tla.vector<256xf8E5M2>
        "tla.store"(%dt, %o) <{operandSegmentSizes = array<i32: 1, 1, 0, 0, 0>}> : (!d_f16_to_e5m2, !tla.vector<256xf8E5M2>) -> ()
      }) {mode = "simd"} : () -> ()
    }) : () -> ()
    return
  }

// CHECK-LABEL: func.func @e5m2_to_bf16
// CHECK: %[[PP0:.*]] = arith.constant 0 : i32
// CHECK-NEXT: "hivm_regbaseintrins.intr.hivm.vcvtff.f8e5m22f32.x"(%{{.*}}, %{{.*}}, %[[PP0]]) : (vector<256xf8E5M2>, vector<256xi1>, i32) -> vector<64xf32>
// CHECK: %[[PP2:.*]] = arith.constant 2 : i32
// CHECK-NEXT: "hivm_regbaseintrins.intr.hivm.vcvtff.f8e5m22f32.x"(%{{.*}}, %{{.*}}, %[[PP2]]) : (vector<256xf8E5M2>, vector<256xi1>, i32) -> vector<64xf32>
// CHECK: ave.hir.vtruncf {{.*}}<part_even>{{.*}} : vector<64xf32>, vector<128xbf16>
// CHECK: ave.hir.vtruncf {{.*}}<part_odd>{{.*}} : vector<64xf32>, vector<128xbf16>
// CHECK: ave.hir.vor
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
        %o = "tla.cast"(%r) <{trait = array<i32: 0, 0, 0>}> : (!tla.vector<256xf8E5M2>) -> !tla.vector<128xbf16>
        "tla.store"(%dt, %o) <{operandSegmentSizes = array<i32: 1, 1, 0, 0, 0>}> : (!d_e5m2_to_bf16, !tla.vector<128xbf16>) -> ()
      }) {mode = "simd"} : () -> ()
    }) : () -> ()
    return
  }

// CHECK-LABEL: func.func @bf16_to_e5m2
// CHECK: ave.hir.vextf {{.*}}<part_even>{{.*}} : vector<128xbf16>, vector<64xf32>
// CHECK: ave.hir.vextf {{.*}}<part_odd>{{.*}} : vector<128xbf16>, vector<64xf32>
// CHECK: arith.constant 0 : i32
// CHECK-NEXT: %[[PP0:.*]] = arith.constant 0 : i32
// CHECK-NEXT: "hivm_regbaseintrins.intr.hivm.vcvtff.f322f8e5m2.x"(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %[[PP0]]) : (vector<64xf32>, vector<256xi1>, i32, i32, i32) -> vector<256xf8E5M2>
// CHECK: %[[PP2:.*]] = arith.constant 2 : i32
// CHECK-NEXT: "hivm_regbaseintrins.intr.hivm.vcvtff.f322f8e5m2.x"(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %[[PP2]]) : (vector<64xf32>, vector<256xi1>, i32, i32, i32) -> vector<256xf8E5M2>
// CHECK: ave.hir.vor
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
        %o = "tla.cast"(%r) <{trait = array<i32: 0, 0, 0>}> : (!tla.vector<128xbf16>) -> !tla.vector<256xf8E5M2>
        "tla.store"(%dt, %o) <{operandSegmentSizes = array<i32: 1, 1, 0, 0, 0>}> : (!d_bf16_to_e5m2, !tla.vector<256xf8E5M2>) -> ()
      }) {mode = "simd"} : () -> ()
    }) : () -> ()
    return
  }
}
