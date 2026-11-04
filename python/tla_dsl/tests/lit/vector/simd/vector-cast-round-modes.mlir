// RUN: %tla_compile %s -o - --mlir-print-ir-after=tla-vector-region-simd 2>&1 | %filecheck %s

// Pin CAST_NONE / CAST_RINT / CAST_ODD trait codes onto HIVM AVE round modes.
// Trait layout: [reg_slot, sat_mode, round_mode]
//   cast_round=0, cast_floor=1, cast_ceil=2, cast_trunc=3,
//   cast_none=4, cast_rint=5, cast_odd=6
//
// CAST_NONE on a same-type (lossy re-round) path must become <rint>, matching
// AscendC. CAST_NONE on a lossless f16->f32 widen still lowers to vextf with
// no round operand. CAST_RINT / CAST_ODD pass through to HIVM RINT / ODD.

!s_f32 = !tla.tensor<!tla.layout<!tla.shape<64>, !tla.stride<1>, !tla.shape<64>, RowMajor>, !tla.coord<0>, !tla.ptr<f32, ub, 4>>
!d_f32 = !tla.tensor<!tla.layout<!tla.shape<64>, !tla.stride<1>, !tla.shape<64>, RowMajor>, !tla.coord<0>, !tla.ptr<f32, ub, 4>>
!s_f16 = !tla.tensor<!tla.layout<!tla.shape<128>, !tla.stride<1>, !tla.shape<128>, RowMajor>, !tla.coord<0>, !tla.ptr<f16, ub, 2>>
!d_f16_as_f32 = !tla.tensor<!tla.layout<!tla.shape<64>, !tla.stride<1>, !tla.shape<64>, RowMajor>, !tla.coord<0>, !tla.ptr<f32, ub, 4>>
!d_i32 = !tla.tensor<!tla.layout<!tla.shape<64>, !tla.stride<1>, !tla.shape<64>, RowMajor>, !tla.coord<0>, !tla.ptr<i32, ub, 4>>

module {
// CHECK-LABEL: func.func @f32_to_f32_none_is_rint
// CHECK: ave.hir.vtrc %{{.*}}, <rint>, %{{.*}} : vector<64xf32>, vector<256xi1>, vector<64xf32>
  func.func @f32_to_f32_none_is_rint(
      %sm: memref<64xf32, #hivm.address_space<ub>>,
      %dm: memref<64xf32, #hivm.address_space<ub>>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %cs = arith.constant 64 : index
    %s = tla.tensor_desc %sm shape [%c1, %cs, %c1, %c1] stride [%cs, %c1, %c1, %c1] origin_shape [%c1, %cs] coord [%c0, %c0] : memref<64xf32, #hivm.address_space<ub>> -> !s_f32
    %d = tla.tensor_desc %dm shape [%c1, %cs, %c1, %c1] stride [%cs, %c1, %c1, %c1] origin_shape [%c1, %cs] coord [%c0, %c0] : memref<64xf32, #hivm.address_space<ub>> -> !d_f32
    "tla.vector"() ({
      "tla.vec.func"() ({
        %ss = "tla.make_shape"() : () -> !tla.shape<64>
        %co = "tla.make_coord"() : () -> !tla.coord<0>
        %st = "tla.tile_view"(%s, %ss, %co) : (!s_f32, !tla.shape<64>, !tla.coord<0>) -> !s_f32
        %dt = "tla.tile_view"(%d, %ss, %co) : (!d_f32, !tla.shape<64>, !tla.coord<0>) -> !d_f32
        %r = "tla.load"(%st) : (!s_f32) -> !tla.vector<64xf32>
        // trait = [zero, nosat, cast_none]
        %o = "tla.cast"(%r) <{trait = array<i32: 0, 2, 4>}> : (!tla.vector<64xf32>) -> !tla.vector<64xf32>
        "tla.store"(%dt, %o) <{operandSegmentSizes = array<i32: 1, 1, 0, 0>}> : (!d_f32, !tla.vector<64xf32>) -> ()
      }) {mode = "simd"} : () -> ()
    }) : () -> ()
    return
  }

// CHECK-LABEL: func.func @f32_to_f32_rint
// CHECK: ave.hir.vtrc %{{.*}}, <rint>, %{{.*}} : vector<64xf32>, vector<256xi1>, vector<64xf32>
  func.func @f32_to_f32_rint(
      %sm: memref<64xf32, #hivm.address_space<ub>>,
      %dm: memref<64xf32, #hivm.address_space<ub>>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %cs = arith.constant 64 : index
    %s = tla.tensor_desc %sm shape [%c1, %cs, %c1, %c1] stride [%cs, %c1, %c1, %c1] origin_shape [%c1, %cs] coord [%c0, %c0] : memref<64xf32, #hivm.address_space<ub>> -> !s_f32
    %d = tla.tensor_desc %dm shape [%c1, %cs, %c1, %c1] stride [%cs, %c1, %c1, %c1] origin_shape [%c1, %cs] coord [%c0, %c0] : memref<64xf32, #hivm.address_space<ub>> -> !d_f32
    "tla.vector"() ({
      "tla.vec.func"() ({
        %ss = "tla.make_shape"() : () -> !tla.shape<64>
        %co = "tla.make_coord"() : () -> !tla.coord<0>
        %st = "tla.tile_view"(%s, %ss, %co) : (!s_f32, !tla.shape<64>, !tla.coord<0>) -> !s_f32
        %dt = "tla.tile_view"(%d, %ss, %co) : (!d_f32, !tla.shape<64>, !tla.coord<0>) -> !d_f32
        %r = "tla.load"(%st) : (!s_f32) -> !tla.vector<64xf32>
        // trait = [zero, nosat, cast_rint]
        %o = "tla.cast"(%r) <{trait = array<i32: 0, 2, 5>}> : (!tla.vector<64xf32>) -> !tla.vector<64xf32>
        "tla.store"(%dt, %o) <{operandSegmentSizes = array<i32: 1, 1, 0, 0>}> : (!d_f32, !tla.vector<64xf32>) -> ()
      }) {mode = "simd"} : () -> ()
    }) : () -> ()
    return
  }

// CAST_ODD is a hardware-supported mode on f32->f16 narrowing (vtruncf), not
// on same-type vtrc / f32->i32.
// CHECK-LABEL: func.func @f32_to_f16_odd
// CHECK: ave.hir.vtruncf {{.*}}<odd>{{.*}} : vector<64xf32>
  func.func @f32_to_f16_odd(
      %sm: memref<64xf32, #hivm.address_space<ub>>,
      %dm: memref<128xf16, #hivm.address_space<ub>>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %cs = arith.constant 64 : index
    %cd = arith.constant 128 : index
    %s = tla.tensor_desc %sm shape [%c1, %cs, %c1, %c1] stride [%cs, %c1, %c1, %c1] origin_shape [%c1, %cs] coord [%c0, %c0] : memref<64xf32, #hivm.address_space<ub>> -> !s_f32
    %d = tla.tensor_desc %dm shape [%c1, %cd, %c1, %c1] stride [%cd, %c1, %c1, %c1] origin_shape [%c1, %cd] coord [%c0, %c0] : memref<128xf16, #hivm.address_space<ub>> -> !s_f16
    "tla.vector"() ({
      "tla.vec.func"() ({
        %ss = "tla.make_shape"() : () -> !tla.shape<64>
        %ds = "tla.make_shape"() : () -> !tla.shape<128>
        %co = "tla.make_coord"() : () -> !tla.coord<0>
        %st = "tla.tile_view"(%s, %ss, %co) : (!s_f32, !tla.shape<64>, !tla.coord<0>) -> !s_f32
        %dt = "tla.tile_view"(%d, %ds, %co) : (!s_f16, !tla.shape<128>, !tla.coord<0>) -> !s_f16
        %r = "tla.load"(%st) : (!s_f32) -> !tla.vector<64xf32>
        // trait = [zero, nosat, cast_odd]
        %o = "tla.cast"(%r) <{trait = array<i32: 0, 2, 6>}> : (!tla.vector<64xf32>) -> !tla.vector<128xf16>
        "tla.store"(%dt, %o) <{operandSegmentSizes = array<i32: 1, 1, 0, 0>}> : (!s_f16, !tla.vector<128xf16>) -> ()
      }) {mode = "simd"} : () -> ()
    }) : () -> ()
    return
  }

// CHECK-LABEL: func.func @f32_to_i32_none_is_rint
// CHECK: ave.hir.vfptosi {{.*}}<rint>{{.*}} : vector<64xf32>
  func.func @f32_to_i32_none_is_rint(
      %sm: memref<64xf32, #hivm.address_space<ub>>,
      %dm: memref<64xi32, #hivm.address_space<ub>>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %cs = arith.constant 64 : index
    %s = tla.tensor_desc %sm shape [%c1, %cs, %c1, %c1] stride [%cs, %c1, %c1, %c1] origin_shape [%c1, %cs] coord [%c0, %c0] : memref<64xf32, #hivm.address_space<ub>> -> !s_f32
    %d = tla.tensor_desc %dm shape [%c1, %cs, %c1, %c1] stride [%cs, %c1, %c1, %c1] origin_shape [%c1, %cs] coord [%c0, %c0] : memref<64xi32, #hivm.address_space<ub>> -> !d_i32
    "tla.vector"() ({
      "tla.vec.func"() ({
        %ss = "tla.make_shape"() : () -> !tla.shape<64>
        %co = "tla.make_coord"() : () -> !tla.coord<0>
        %st = "tla.tile_view"(%s, %ss, %co) : (!s_f32, !tla.shape<64>, !tla.coord<0>) -> !s_f32
        %dt = "tla.tile_view"(%d, %ss, %co) : (!d_i32, !tla.shape<64>, !tla.coord<0>) -> !d_i32
        %r = "tla.load"(%st) : (!s_f32) -> !tla.vector<64xf32>
        // trait = [zero, nosat, cast_none]
        %o = "tla.cast"(%r) <{trait = array<i32: 0, 2, 4>}> : (!tla.vector<64xf32>) -> !tla.vector<64xi32>
        "tla.store"(%dt, %o) <{operandSegmentSizes = array<i32: 1, 1, 0, 0>}> : (!d_i32, !tla.vector<64xi32>) -> ()
      }) {mode = "simd"} : () -> ()
    }) : () -> ()
    return
  }

// Lossless widen: CAST_NONE is accepted and the round operand is dropped.
// CHECK-LABEL: func.func @f16_to_f32_none_is_vextf
// CHECK: ave.hir.vextf
// CHECK-NOT: ave.hir.vtrc
  func.func @f16_to_f32_none_is_vextf(
      %sm: memref<128xf16, #hivm.address_space<ub>>,
      %dm: memref<64xf32, #hivm.address_space<ub>>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %cs = arith.constant 128 : index
    %cd = arith.constant 64 : index
    %s = tla.tensor_desc %sm shape [%c1, %cs, %c1, %c1] stride [%cs, %c1, %c1, %c1] origin_shape [%c1, %cs] coord [%c0, %c0] : memref<128xf16, #hivm.address_space<ub>> -> !s_f16
    %d = tla.tensor_desc %dm shape [%c1, %cd, %c1, %c1] stride [%cd, %c1, %c1, %c1] origin_shape [%c1, %cd] coord [%c0, %c0] : memref<64xf32, #hivm.address_space<ub>> -> !d_f16_as_f32
    "tla.vector"() ({
      "tla.vec.func"() ({
        %ss = "tla.make_shape"() : () -> !tla.shape<128>
        %ds = "tla.make_shape"() : () -> !tla.shape<64>
        %co = "tla.make_coord"() : () -> !tla.coord<0>
        %st = "tla.tile_view"(%s, %ss, %co) : (!s_f16, !tla.shape<128>, !tla.coord<0>) -> !s_f16
        %dt = "tla.tile_view"(%d, %ds, %co) : (!d_f16_as_f32, !tla.shape<64>, !tla.coord<0>) -> !d_f16_as_f32
        %r = "tla.load"(%st) : (!s_f16) -> !tla.vector<128xf16>
        // trait = [zero, nosat, cast_none]; vextf has no rnd operand
        %o = "tla.cast"(%r) <{trait = array<i32: 0, 2, 4>}> : (!tla.vector<128xf16>) -> !tla.vector<64xf32>
        "tla.store"(%dt, %o) <{operandSegmentSizes = array<i32: 1, 1, 0, 0>}> : (!d_f16_as_f32, !tla.vector<64xf32>) -> ()
      }) {mode = "simd"} : () -> ()
    }) : () -> ()
    return
  }
}
