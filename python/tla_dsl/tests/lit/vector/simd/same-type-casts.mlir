// RUN: %tla_compile %s -o - --mlir-print-ir-after=tla-vector-region-simd 2>&1 | %filecheck %s --implicit-check-not=ave.hir.vextf --implicit-check-not=ave.hir.vtruncf

// Same-type float casts: f16->f16, bf16->bf16, f32->f32.
//
// There is no width to convert, so each one lowers to the AVE vtrc op --
// "just rounding": the round mode reaches the instruction and nothing else
// does. That is worth pinning precisely, because the generic float-float
// dispatch right below the same-type check would otherwise pick vextf, which
// HIVMAVEToAVEIntrin cannot lower when src and dst element types match and
// dies with a fatal "Unsupported type conversion op." rather than a
// diagnostic. Each case therefore CHECKs the vtrc instruction and CHECK-NOTs
// the vextf the old path emitted.
//
// Each case uses SatMode.UNKNOWN and checks a distinct rounding mode.
//
// The vector types stay exact for the same reason the unsigned cast tests pin
// them: a vtrc at the wrong width would still typecheck. The lowered dump
// carries the round mode as a plain enum attribute, so each case pins its
// spelling -- <round>, <trunc>, <floor> -- and a trait code that stopped
// reaching the op would fail here.

!s_f32 = !tla.tensor<!tla.layout<!tla.shape<64>, !tla.stride<1>, !tla.shape<64>, RowMajor>, !tla.coord<0>, !tla.ptr<f32, ub, 4>>
!d_f32 = !tla.tensor<!tla.layout<!tla.shape<64>, !tla.stride<1>, !tla.shape<64>, RowMajor>, !tla.coord<0>, !tla.ptr<f32, ub, 4>>
!s_f16 = !tla.tensor<!tla.layout<!tla.shape<128>, !tla.stride<1>, !tla.shape<128>, RowMajor>, !tla.coord<0>, !tla.ptr<f16, ub, 2>>
!d_f16 = !tla.tensor<!tla.layout<!tla.shape<128>, !tla.stride<1>, !tla.shape<128>, RowMajor>, !tla.coord<0>, !tla.ptr<f16, ub, 2>>
!s_bf16 = !tla.tensor<!tla.layout<!tla.shape<128>, !tla.stride<1>, !tla.shape<128>, RowMajor>, !tla.coord<0>, !tla.ptr<bf16, ub, 2>>
!d_bf16 = !tla.tensor<!tla.layout<!tla.shape<128>, !tla.stride<1>, !tla.shape<128>, RowMajor>, !tla.coord<0>, !tla.ptr<bf16, ub, 2>>

module {
// f32 -> f32, round to nearest. vtrc has no saturation operand or part selector.
// CHECK-LABEL: func.func @f32_to_f32_round
// CHECK: ave.hir.vtrc %{{.*}}, <round>, %{{.*}} : vector<64xf32>, vector<256xi1>, vector<64xf32>
  func.func @f32_to_f32_round(
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
        // trait = [reg_slot, sat_mode, round_mode] = [one, unknown, cast_round]
        %o = "tla.cast"(%r) <{trait = array<i32: 1, 0, 0>}> : (!tla.vector<64xf32>) -> !tla.vector<64xf32>
        "tla.store"(%dt, %o) <{operandSegmentSizes = array<i32: 1, 1, 0, 0, 0>}> : (!d_f32, !tla.vector<64xf32>) -> ()
      }) {mode = "simd"} : () -> ()
    }) : () -> ()
    return
  }

// f16 -> f16, round toward zero: the integer-valued f32 idiom, spelled in f16.
// CHECK-LABEL: func.func @f16_to_f16_trunc
// CHECK: ave.hir.vtrc %{{.*}}, <trunc>, %{{.*}} : vector<128xf16>, vector<256xi1>, vector<128xf16>
  func.func @f16_to_f16_trunc(
      %sm: memref<128xf16, #hivm.address_space<ub>>,
      %dm: memref<128xf16, #hivm.address_space<ub>>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %cs = arith.constant 128 : index
    %s = tla.tensor_desc %sm shape [%c1, %cs, %c1, %c1] stride [%cs, %c1, %c1, %c1] origin_shape [%c1, %cs] coord [%c0, %c0] : memref<128xf16, #hivm.address_space<ub>> -> !s_f16
    %d = tla.tensor_desc %dm shape [%c1, %cs, %c1, %c1] stride [%cs, %c1, %c1, %c1] origin_shape [%c1, %cs] coord [%c0, %c0] : memref<128xf16, #hivm.address_space<ub>> -> !d_f16
    "tla.vector"() ({
      "tla.vec.func"() ({
        %ss = "tla.make_shape"() : () -> !tla.shape<128>
        %co = "tla.make_coord"() : () -> !tla.coord<0>
        %st = "tla.tile_view"(%s, %ss, %co) : (!s_f16, !tla.shape<128>, !tla.coord<0>) -> !s_f16
        %dt = "tla.tile_view"(%d, %ss, %co) : (!d_f16, !tla.shape<128>, !tla.coord<0>) -> !d_f16
        %r = "tla.load"(%st) : (!s_f16) -> !tla.vector<128xf16>
        // trait = [reg_slot, sat_mode, round_mode] = [zero, unknown, cast_trunc]
        %o = "tla.cast"(%r) <{trait = array<i32: 0, 0, 3>}> : (!tla.vector<128xf16>) -> !tla.vector<128xf16>
        "tla.store"(%dt, %o) <{operandSegmentSizes = array<i32: 1, 1, 0, 0, 0>}> : (!d_f16, !tla.vector<128xf16>) -> ()
      }) {mode = "simd"} : () -> ()
    }) : () -> ()
    return
  }

// bf16 -> bf16, round toward minus infinity.
// CHECK-LABEL: func.func @bf16_to_bf16_floor
// CHECK: ave.hir.vtrc %{{.*}}, <floor>, %{{.*}} : vector<128xbf16>, vector<256xi1>, vector<128xbf16>
  func.func @bf16_to_bf16_floor(
      %sm: memref<128xbf16, #hivm.address_space<ub>>,
      %dm: memref<128xbf16, #hivm.address_space<ub>>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %cs = arith.constant 128 : index
    %s = tla.tensor_desc %sm shape [%c1, %cs, %c1, %c1] stride [%cs, %c1, %c1, %c1] origin_shape [%c1, %cs] coord [%c0, %c0] : memref<128xbf16, #hivm.address_space<ub>> -> !s_bf16
    %d = tla.tensor_desc %dm shape [%c1, %cs, %c1, %c1] stride [%cs, %c1, %c1, %c1] origin_shape [%c1, %cs] coord [%c0, %c0] : memref<128xbf16, #hivm.address_space<ub>> -> !d_bf16
    "tla.vector"() ({
      "tla.vec.func"() ({
        %ss = "tla.make_shape"() : () -> !tla.shape<128>
        %co = "tla.make_coord"() : () -> !tla.coord<0>
        %st = "tla.tile_view"(%s, %ss, %co) : (!s_bf16, !tla.shape<128>, !tla.coord<0>) -> !s_bf16
        %dt = "tla.tile_view"(%d, %ss, %co) : (!d_bf16, !tla.shape<128>, !tla.coord<0>) -> !d_bf16
        %r = "tla.load"(%st) : (!s_bf16) -> !tla.vector<128xbf16>
        // trait = [reg_slot, sat_mode, round_mode] = [zero, unknown, cast_floor]
        %o = "tla.cast"(%r) <{trait = array<i32: 0, 0, 1>}> : (!tla.vector<128xbf16>) -> !tla.vector<128xbf16>
        "tla.store"(%dt, %o) <{operandSegmentSizes = array<i32: 1, 1, 0, 0, 0>}> : (!d_bf16, !tla.vector<128xbf16>) -> ()
      }) {mode = "simd"} : () -> ()
    }) : () -> ()
    return
  }
}
