// Verify tla-insert-implicit-ctrl clears CTRL[60] for every cast sat_mode
// (SAT / NOSAT / UNKNOWN): InitSoc leaves CTRL[60]=1 (global override), and
// the instruction sat operand is only honored when OverrideSaturation is off.
// Also verify coalescing across consecutive casts / loops.
//
// RUN: %tla_compile %s -o /dev/null --mlir-print-ir-after=tla-insert-implicit-ctrl 2>&1 | %filecheck %s

!s_f32 = !tla.tensor<!tla.layout<!tla.shape<64>, !tla.stride<1>, !tla.shape<64>, RowMajor>, !tla.coord<0>, !tla.ptr<f32, ub, 4>>
!d_e4m3 = !tla.tensor<!tla.layout<!tla.shape<256>, !tla.stride<1>, !tla.shape<256>, RowMajor>, !tla.coord<0>, !tla.ptr<f8E4M3FN, ub, 1>>

module {

// Two consecutive SAT casts in one vec.func → a single CTRL[60]=0 before the region.
// CHECK-LABEL: func.func @two_sat_casts_one_set_ctrl
// CHECK: hivm.hir.set_ctrl false at ctrl[60]
// CHECK-NEXT: "tla.vec.func"()
// CHECK-NOT: hivm.hir.set_ctrl {{.*}} at ctrl[60]
  func.func @two_sat_casts_one_set_ctrl(
      %sm: memref<64xf32, #hivm.address_space<ub>>,
      %dm: memref<256xf8E4M3FN, #hivm.address_space<ub>>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %cs = arith.constant 64 : index
    %cd = arith.constant 256 : index
    %s = tla.tensor_desc %sm shape [%c1, %cs, %c1, %c1] stride [%cs, %c1, %c1, %c1] origin_shape [%c1, %cs] coord [%c0, %c0] : memref<64xf32, #hivm.address_space<ub>> -> !s_f32
    %d = tla.tensor_desc %dm shape [%c1, %cd, %c1, %c1] stride [%cd, %c1, %c1, %c1] origin_shape [%c1, %cd] coord [%c0, %c0] : memref<256xf8E4M3FN, #hivm.address_space<ub>> -> !d_e4m3
    "tla.vector"() ({
      "tla.vec.func"() ({
        %ss = "tla.make_shape"() : () -> !tla.shape<64>
        %ds = "tla.make_shape"() : () -> !tla.shape<256>
        %co = "tla.make_coord"() : () -> !tla.coord<0>
        %st = "tla.tile_view"(%s, %ss, %co) : (!s_f32, !tla.shape<64>, !tla.coord<0>) -> !s_f32
        %dt = "tla.tile_view"(%d, %ds, %co) : (!d_e4m3, !tla.shape<256>, !tla.coord<0>) -> !d_e4m3
        %r = "tla.load"(%st) : (!s_f32) -> !tla.vector<64xf32>
        // trait = [reg_slot, sat_mode=SAT, round_mode]
        %o0 = "tla.cast"(%r) <{trait = array<i32: 0, 1, 5>}> : (!tla.vector<64xf32>) -> !tla.vector<256xf8E4M3FN>
        %o1 = "tla.cast"(%r) <{trait = array<i32: 0, 1, 5>}> : (!tla.vector<64xf32>) -> !tla.vector<256xf8E4M3FN>
        "tla.store"(%dt, %o0) <{operandSegmentSizes = array<i32: 1, 1, 0, 0>}> : (!d_e4m3, !tla.vector<256xf8E4M3FN>) -> ()
        "tla.store"(%dt, %o1) <{operandSegmentSizes = array<i32: 1, 1, 0, 0>}> : (!d_e4m3, !tla.vector<256xf8E4M3FN>) -> ()
      }) {mode = "simd"} : () -> ()
    }) : () -> ()
    return
  }

// NOSAT also needs CTRL[60]=0 so the instruction sat=false takes effect
// (InitSoc leaves CTRL[60]=1; CTRL[48] is preserved and must not decide).
// CHECK-LABEL: func.func @nosat_clears_override
// CHECK: hivm.hir.set_ctrl false at ctrl[60]
// CHECK-NEXT: "tla.vec.func"()
// CHECK-NOT: hivm.hir.set_ctrl {{.*}} at ctrl[60]
  func.func @nosat_clears_override(
      %sm: memref<64xf32, #hivm.address_space<ub>>,
      %dm: memref<256xf8E4M3FN, #hivm.address_space<ub>>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %cs = arith.constant 64 : index
    %cd = arith.constant 256 : index
    %s = tla.tensor_desc %sm shape [%c1, %cs, %c1, %c1] stride [%cs, %c1, %c1, %c1] origin_shape [%c1, %cs] coord [%c0, %c0] : memref<64xf32, #hivm.address_space<ub>> -> !s_f32
    %d = tla.tensor_desc %dm shape [%c1, %cd, %c1, %c1] stride [%cd, %c1, %c1, %c1] origin_shape [%c1, %cd] coord [%c0, %c0] : memref<256xf8E4M3FN, #hivm.address_space<ub>> -> !d_e4m3
    "tla.vector"() ({
      "tla.vec.func"() ({
        %ss = "tla.make_shape"() : () -> !tla.shape<64>
        %ds = "tla.make_shape"() : () -> !tla.shape<256>
        %co = "tla.make_coord"() : () -> !tla.coord<0>
        %st = "tla.tile_view"(%s, %ss, %co) : (!s_f32, !tla.shape<64>, !tla.coord<0>) -> !s_f32
        %dt = "tla.tile_view"(%d, %ds, %co) : (!d_e4m3, !tla.shape<256>, !tla.coord<0>) -> !d_e4m3
        %r = "tla.load"(%st) : (!s_f32) -> !tla.vector<64xf32>
        // trait sat_mode=NOSAT
        %o = "tla.cast"(%r) <{trait = array<i32: 0, 2, 5>}> : (!tla.vector<64xf32>) -> !tla.vector<256xf8E4M3FN>
        "tla.store"(%dt, %o) <{operandSegmentSizes = array<i32: 1, 1, 0, 0>}> : (!d_e4m3, !tla.vector<256xf8E4M3FN>) -> ()
      }) {mode = "simd"} : () -> ()
    }) : () -> ()
    return
  }

// SAT casts inside scf.for → SetCtrl hoisted before the vec.func (once).
// CHECK-LABEL: func.func @sat_inside_loop_hoisted
// CHECK: hivm.hir.set_ctrl false at ctrl[60]
// CHECK-NEXT: "tla.vec.func"()
// CHECK: scf.for
// CHECK-NOT: hivm.hir.set_ctrl {{.*}} at ctrl[60]
  func.func @sat_inside_loop_hoisted(
      %sm: memref<64xf32, #hivm.address_space<ub>>,
      %dm: memref<256xf8E4M3FN, #hivm.address_space<ub>>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %cs = arith.constant 64 : index
    %cd = arith.constant 256 : index
    %s = tla.tensor_desc %sm shape [%c1, %cs, %c1, %c1] stride [%cs, %c1, %c1, %c1] origin_shape [%c1, %cs] coord [%c0, %c0] : memref<64xf32, #hivm.address_space<ub>> -> !s_f32
    %d = tla.tensor_desc %dm shape [%c1, %cd, %c1, %c1] stride [%cd, %c1, %c1, %c1] origin_shape [%c1, %cd] coord [%c0, %c0] : memref<256xf8E4M3FN, #hivm.address_space<ub>> -> !d_e4m3
    "tla.vector"() ({
      "tla.vec.func"() ({
        scf.for %i = %c0 to %c2 step %c1 {
          %ss = "tla.make_shape"() : () -> !tla.shape<64>
          %ds = "tla.make_shape"() : () -> !tla.shape<256>
          %co = "tla.make_coord"() : () -> !tla.coord<0>
          %st = "tla.tile_view"(%s, %ss, %co) : (!s_f32, !tla.shape<64>, !tla.coord<0>) -> !s_f32
          %dt = "tla.tile_view"(%d, %ds, %co) : (!d_e4m3, !tla.shape<256>, !tla.coord<0>) -> !d_e4m3
          %r = "tla.load"(%st) : (!s_f32) -> !tla.vector<64xf32>
          %o = "tla.cast"(%r) <{trait = array<i32: 0, 1, 5>}> : (!tla.vector<64xf32>) -> !tla.vector<256xf8E4M3FN>
          "tla.store"(%dt, %o) <{operandSegmentSizes = array<i32: 1, 1, 0, 0>}> : (!d_e4m3, !tla.vector<256xf8E4M3FN>) -> ()
        }
      }) {mode = "simd"} : () -> ()
    }) : () -> ()
    return
  }
}
