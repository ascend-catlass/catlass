// RUN: %tla_compile %s -o - --mlir-print-ir-after=tla-vector-region-simd 2>&1 | %filecheck %s

// Every unsigned cast route the vector unit accepts, and the instruction each
// one selects.
//
// The point of pinning the op and its attributes is that the signed and
// unsigned forms differ only in a letter or in one attribute, and picking the
// signed one is silently wrong rather than an error: an s8->s32 widen
// sign-extends, so every u8 value >= 128 -- half the range, and every OCP e8m0
// block exponent above 2^0 -- comes out as a large negative number that still
// looks like a plausible integer. Each widen and each int->float case therefore
// CHECK-NOTs its signed twin, and each narrow pins `uni = ui2ui`, which is what
// selects VcvtiiU322U8 over VcvtiiS322U8 further down.
//
// The vector types in each CHECK are signless: every AVE op is typed on
// signless integers, so a u8 VectorSSA is carried as vector<Nxi8> inside the
// region and the signedness lives in the op choice and the `uni` attribute.
// Pinning the types therefore also pins the widths, so a route that selected
// the right op at the wrong width still fails here.
//
// There is deliberately no u64 case. vtrunci has a 64->32 arm, but nothing can
// feed it an unsigned operand: there is no 64-bit tile element type to load a
// u64 vector from, and no widen yields one either, since vextui has no
// U32 -> U64 arm. A lit case would prove only that hand-written MLIR can reach
// an instruction no DSL caller can, which is why the lowering now refuses a
// 64-bit unsigned end outright. (i64 differs -- unreachable as a tile, but
// reachable as a register value through the signed i32 -> i64 widen.)
//
// u8 -> u32 and u32 -> u8 are 4x steps and carry the pack pattern pp0..pp3
// rather than the 2x even/odd part, so one case runs with reg_slot THREE to
// keep the quarter-selecting form covered.


!s_u8_to_u16 = !tla.tensor<!tla.layout<!tla.shape<256>, !tla.stride<1>, !tla.shape<256>, RowMajor>, !tla.coord<0>, !tla.ptr<ui8, ub, 1>>
!d_u8_to_u16 = !tla.tensor<!tla.layout<!tla.shape<128>, !tla.stride<1>, !tla.shape<128>, RowMajor>, !tla.coord<0>, !tla.ptr<ui16, ub, 2>>
!s_u16_to_u32 = !tla.tensor<!tla.layout<!tla.shape<128>, !tla.stride<1>, !tla.shape<128>, RowMajor>, !tla.coord<0>, !tla.ptr<ui16, ub, 2>>
!d_u16_to_u32 = !tla.tensor<!tla.layout<!tla.shape<64>, !tla.stride<1>, !tla.shape<64>, RowMajor>, !tla.coord<0>, !tla.ptr<ui32, ub, 4>>
!s_u8_to_u32 = !tla.tensor<!tla.layout<!tla.shape<256>, !tla.stride<1>, !tla.shape<256>, RowMajor>, !tla.coord<0>, !tla.ptr<ui8, ub, 1>>
!d_u8_to_u32 = !tla.tensor<!tla.layout<!tla.shape<64>, !tla.stride<1>, !tla.shape<64>, RowMajor>, !tla.coord<0>, !tla.ptr<ui32, ub, 4>>
!s_u8_to_u32_slot_three = !tla.tensor<!tla.layout<!tla.shape<256>, !tla.stride<1>, !tla.shape<256>, RowMajor>, !tla.coord<0>, !tla.ptr<ui8, ub, 1>>
!d_u8_to_u32_slot_three = !tla.tensor<!tla.layout<!tla.shape<64>, !tla.stride<1>, !tla.shape<64>, RowMajor>, !tla.coord<0>, !tla.ptr<ui32, ub, 4>>
!s_u16_to_u8 = !tla.tensor<!tla.layout<!tla.shape<128>, !tla.stride<1>, !tla.shape<128>, RowMajor>, !tla.coord<0>, !tla.ptr<ui16, ub, 2>>
!d_u16_to_u8 = !tla.tensor<!tla.layout<!tla.shape<256>, !tla.stride<1>, !tla.shape<256>, RowMajor>, !tla.coord<0>, !tla.ptr<ui8, ub, 1>>
!s_u32_to_u16 = !tla.tensor<!tla.layout<!tla.shape<64>, !tla.stride<1>, !tla.shape<64>, RowMajor>, !tla.coord<0>, !tla.ptr<ui32, ub, 4>>
!d_u32_to_u16 = !tla.tensor<!tla.layout<!tla.shape<128>, !tla.stride<1>, !tla.shape<128>, RowMajor>, !tla.coord<0>, !tla.ptr<ui16, ub, 2>>
!s_u32_to_u8 = !tla.tensor<!tla.layout<!tla.shape<64>, !tla.stride<1>, !tla.shape<64>, RowMajor>, !tla.coord<0>, !tla.ptr<ui32, ub, 4>>
!d_u32_to_u8 = !tla.tensor<!tla.layout<!tla.shape<256>, !tla.stride<1>, !tla.shape<256>, RowMajor>, !tla.coord<0>, !tla.ptr<ui8, ub, 1>>
!s_u8_to_f16 = !tla.tensor<!tla.layout<!tla.shape<256>, !tla.stride<1>, !tla.shape<256>, RowMajor>, !tla.coord<0>, !tla.ptr<ui8, ub, 1>>
!d_u8_to_f16 = !tla.tensor<!tla.layout<!tla.shape<128>, !tla.stride<1>, !tla.shape<128>, RowMajor>, !tla.coord<0>, !tla.ptr<f16, ub, 2>>
!s_f16_to_u8 = !tla.tensor<!tla.layout<!tla.shape<128>, !tla.stride<1>, !tla.shape<128>, RowMajor>, !tla.coord<0>, !tla.ptr<f16, ub, 2>>
!d_f16_to_u8 = !tla.tensor<!tla.layout<!tla.shape<256>, !tla.stride<1>, !tla.shape<256>, RowMajor>, !tla.coord<0>, !tla.ptr<ui8, ub, 1>>
// Unsigned max/min/reduce: same signed-twin risk once AVE vectors are signless.
!u16_vec = !tla.tensor<!tla.layout<!tla.shape<128>, !tla.stride<1>, !tla.shape<128>, RowMajor>, !tla.coord<0>, !tla.ptr<ui16, ub, 2>>
!u16_vec1 = !tla.tensor<!tla.layout<!tla.shape<1>, !tla.stride<1>, !tla.shape<1>, RowMajor>, !tla.coord<0>, !tla.ptr<ui16, ub, 2>>

!s_u8_to_f32 = !tla.tensor<!tla.layout<!tla.shape<256>, !tla.stride<1>, !tla.shape<256>, RowMajor>, !tla.coord<0>, !tla.ptr<ui8, ub, 1>>
!d_u8_to_f32 = !tla.tensor<!tla.layout<!tla.shape<64>, !tla.stride<1>, !tla.shape<64>, RowMajor>, !tla.coord<0>, !tla.ptr<f32, ub, 4>>

module {
// CHECK-LABEL: func.func @u8_to_u16
// CHECK: ave.hir.vextui %{{.*}} {part = #ave.vcvt_part_type<part_even>} : vector<256xi8>, vector<128xi16>
// CHECK-NOT: ave.hir.vextsi
  func.func @u8_to_u16(
      %sm: memref<256xui8, #hivm.address_space<ub>>,
      %dm: memref<128xui16, #hivm.address_space<ub>>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %cs = arith.constant 256 : index
    %cd = arith.constant 128 : index
    %s = tla.tensor_desc %sm shape [%c1, %cs, %c1, %c1] stride [%cs, %c1, %c1, %c1] origin_shape [%c1, %cs] coord [%c0, %c0] : memref<256xui8, #hivm.address_space<ub>> -> !s_u8_to_u16
    %d = tla.tensor_desc %dm shape [%c1, %cd, %c1, %c1] stride [%cd, %c1, %c1, %c1] origin_shape [%c1, %cd] coord [%c0, %c0] : memref<128xui16, #hivm.address_space<ub>> -> !d_u8_to_u16
    "tla.vector"() ({
      "tla.vec.func"() ({
        %ss = "tla.make_shape"() : () -> !tla.shape<256>
        %ds = "tla.make_shape"() : () -> !tla.shape<128>
        %co = "tla.make_coord"() : () -> !tla.coord<0>
        %st = "tla.tile_view"(%s, %ss, %co) : (!s_u8_to_u16, !tla.shape<256>, !tla.coord<0>) -> !s_u8_to_u16
        %dt = "tla.tile_view"(%d, %ds, %co) : (!d_u8_to_u16, !tla.shape<128>, !tla.coord<0>) -> !d_u8_to_u16
        %r = "tla.load"(%st) : (!s_u8_to_u16) -> !tla.vector<256xui8>
        // trait = [reg_slot, sat_mode, round_mode]
        %o = "tla.cast"(%r) <{trait = array<i32: 0, 0, 4>}> : (!tla.vector<256xui8>) -> !tla.vector<128xui16>
        "tla.store"(%dt, %o) <{operandSegmentSizes = array<i32: 1, 1, 0, 0, 0>}> : (!d_u8_to_u16, !tla.vector<128xui16>) -> ()
      }) {mode = "simd"} : () -> ()
    }) : () -> ()
    return
  }

// CHECK-LABEL: func.func @u16_to_u32
// CHECK: ave.hir.vextui %{{.*}} {part = #ave.vcvt_part_type<part_even>} : vector<128xi16>, vector<64xi32>
// CHECK-NOT: ave.hir.vextsi
  func.func @u16_to_u32(
      %sm: memref<128xui16, #hivm.address_space<ub>>,
      %dm: memref<64xui32, #hivm.address_space<ub>>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %cs = arith.constant 128 : index
    %cd = arith.constant 64 : index
    %s = tla.tensor_desc %sm shape [%c1, %cs, %c1, %c1] stride [%cs, %c1, %c1, %c1] origin_shape [%c1, %cs] coord [%c0, %c0] : memref<128xui16, #hivm.address_space<ub>> -> !s_u16_to_u32
    %d = tla.tensor_desc %dm shape [%c1, %cd, %c1, %c1] stride [%cd, %c1, %c1, %c1] origin_shape [%c1, %cd] coord [%c0, %c0] : memref<64xui32, #hivm.address_space<ub>> -> !d_u16_to_u32
    "tla.vector"() ({
      "tla.vec.func"() ({
        %ss = "tla.make_shape"() : () -> !tla.shape<128>
        %ds = "tla.make_shape"() : () -> !tla.shape<64>
        %co = "tla.make_coord"() : () -> !tla.coord<0>
        %st = "tla.tile_view"(%s, %ss, %co) : (!s_u16_to_u32, !tla.shape<128>, !tla.coord<0>) -> !s_u16_to_u32
        %dt = "tla.tile_view"(%d, %ds, %co) : (!d_u16_to_u32, !tla.shape<64>, !tla.coord<0>) -> !d_u16_to_u32
        %r = "tla.load"(%st) : (!s_u16_to_u32) -> !tla.vector<128xui16>
        // trait = [reg_slot, sat_mode, round_mode]
        %o = "tla.cast"(%r) <{trait = array<i32: 0, 0, 4>}> : (!tla.vector<128xui16>) -> !tla.vector<64xui32>
        "tla.store"(%dt, %o) <{operandSegmentSizes = array<i32: 1, 1, 0, 0, 0>}> : (!d_u16_to_u32, !tla.vector<64xui32>) -> ()
      }) {mode = "simd"} : () -> ()
    }) : () -> ()
    return
  }

// CHECK-LABEL: func.func @u8_to_u32
// CHECK: ave.hir.vextui %{{.*}} {pp = #ave.vcvt_pp_type<pp0>} : vector<256xi8>, vector<64xi32>
// CHECK-NOT: ave.hir.vextsi
  func.func @u8_to_u32(
      %sm: memref<256xui8, #hivm.address_space<ub>>,
      %dm: memref<64xui32, #hivm.address_space<ub>>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %cs = arith.constant 256 : index
    %cd = arith.constant 64 : index
    %s = tla.tensor_desc %sm shape [%c1, %cs, %c1, %c1] stride [%cs, %c1, %c1, %c1] origin_shape [%c1, %cs] coord [%c0, %c0] : memref<256xui8, #hivm.address_space<ub>> -> !s_u8_to_u32
    %d = tla.tensor_desc %dm shape [%c1, %cd, %c1, %c1] stride [%cd, %c1, %c1, %c1] origin_shape [%c1, %cd] coord [%c0, %c0] : memref<64xui32, #hivm.address_space<ub>> -> !d_u8_to_u32
    "tla.vector"() ({
      "tla.vec.func"() ({
        %ss = "tla.make_shape"() : () -> !tla.shape<256>
        %ds = "tla.make_shape"() : () -> !tla.shape<64>
        %co = "tla.make_coord"() : () -> !tla.coord<0>
        %st = "tla.tile_view"(%s, %ss, %co) : (!s_u8_to_u32, !tla.shape<256>, !tla.coord<0>) -> !s_u8_to_u32
        %dt = "tla.tile_view"(%d, %ds, %co) : (!d_u8_to_u32, !tla.shape<64>, !tla.coord<0>) -> !d_u8_to_u32
        %r = "tla.load"(%st) : (!s_u8_to_u32) -> !tla.vector<256xui8>
        // trait = [reg_slot, sat_mode, round_mode]
        %o = "tla.cast"(%r) <{trait = array<i32: 0, 0, 4>}> : (!tla.vector<256xui8>) -> !tla.vector<64xui32>
        "tla.store"(%dt, %o) <{operandSegmentSizes = array<i32: 1, 1, 0, 0, 0>}> : (!d_u8_to_u32, !tla.vector<64xui32>) -> ()
      }) {mode = "simd"} : () -> ()
    }) : () -> ()
    return
  }

// CHECK-LABEL: func.func @u8_to_u32_slot_three
// CHECK: ave.hir.vextui %{{.*}} {pp = #ave.vcvt_pp_type<pp3>} : vector<256xi8>, vector<64xi32>
// CHECK-NOT: ave.hir.vextsi
  func.func @u8_to_u32_slot_three(
      %sm: memref<256xui8, #hivm.address_space<ub>>,
      %dm: memref<64xui32, #hivm.address_space<ub>>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %cs = arith.constant 256 : index
    %cd = arith.constant 64 : index
    %s = tla.tensor_desc %sm shape [%c1, %cs, %c1, %c1] stride [%cs, %c1, %c1, %c1] origin_shape [%c1, %cs] coord [%c0, %c0] : memref<256xui8, #hivm.address_space<ub>> -> !s_u8_to_u32_slot_three
    %d = tla.tensor_desc %dm shape [%c1, %cd, %c1, %c1] stride [%cd, %c1, %c1, %c1] origin_shape [%c1, %cd] coord [%c0, %c0] : memref<64xui32, #hivm.address_space<ub>> -> !d_u8_to_u32_slot_three
    "tla.vector"() ({
      "tla.vec.func"() ({
        %ss = "tla.make_shape"() : () -> !tla.shape<256>
        %ds = "tla.make_shape"() : () -> !tla.shape<64>
        %co = "tla.make_coord"() : () -> !tla.coord<0>
        %st = "tla.tile_view"(%s, %ss, %co) : (!s_u8_to_u32_slot_three, !tla.shape<256>, !tla.coord<0>) -> !s_u8_to_u32_slot_three
        %dt = "tla.tile_view"(%d, %ds, %co) : (!d_u8_to_u32_slot_three, !tla.shape<64>, !tla.coord<0>) -> !d_u8_to_u32_slot_three
        %r = "tla.load"(%st) : (!s_u8_to_u32_slot_three) -> !tla.vector<256xui8>
        // trait = [reg_slot, sat_mode, round_mode]
        %o = "tla.cast"(%r) <{trait = array<i32: 3, 0, 4>}> : (!tla.vector<256xui8>) -> !tla.vector<64xui32>
        "tla.store"(%dt, %o) <{operandSegmentSizes = array<i32: 1, 1, 0, 0, 0>}> : (!d_u8_to_u32_slot_three, !tla.vector<64xui32>) -> ()
      }) {mode = "simd"} : () -> ()
    }) : () -> ()
    return
  }

// CHECK-LABEL: func.func @u16_to_u8
// CHECK: ave.hir.vtrunci %{{.*}} {part = #ave.vcvt_part_type<part_even>, uni = #hivm.unsigned_mode<ui2ui>} : vector<128xi16>, vector<256xi8>
  func.func @u16_to_u8(
      %sm: memref<128xui16, #hivm.address_space<ub>>,
      %dm: memref<256xui8, #hivm.address_space<ub>>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %cs = arith.constant 128 : index
    %cd = arith.constant 256 : index
    %s = tla.tensor_desc %sm shape [%c1, %cs, %c1, %c1] stride [%cs, %c1, %c1, %c1] origin_shape [%c1, %cs] coord [%c0, %c0] : memref<128xui16, #hivm.address_space<ub>> -> !s_u16_to_u8
    %d = tla.tensor_desc %dm shape [%c1, %cd, %c1, %c1] stride [%cd, %c1, %c1, %c1] origin_shape [%c1, %cd] coord [%c0, %c0] : memref<256xui8, #hivm.address_space<ub>> -> !d_u16_to_u8
    "tla.vector"() ({
      "tla.vec.func"() ({
        %ss = "tla.make_shape"() : () -> !tla.shape<128>
        %ds = "tla.make_shape"() : () -> !tla.shape<256>
        %co = "tla.make_coord"() : () -> !tla.coord<0>
        %st = "tla.tile_view"(%s, %ss, %co) : (!s_u16_to_u8, !tla.shape<128>, !tla.coord<0>) -> !s_u16_to_u8
        %dt = "tla.tile_view"(%d, %ds, %co) : (!d_u16_to_u8, !tla.shape<256>, !tla.coord<0>) -> !d_u16_to_u8
        %r = "tla.load"(%st) : (!s_u16_to_u8) -> !tla.vector<128xui16>
        // trait = [reg_slot, sat_mode, round_mode]
        %o = "tla.cast"(%r) <{trait = array<i32: 0, 2, 4>}> : (!tla.vector<128xui16>) -> !tla.vector<256xui8>
        "tla.store"(%dt, %o) <{operandSegmentSizes = array<i32: 1, 1, 0, 0, 0>}> : (!d_u16_to_u8, !tla.vector<256xui8>) -> ()
      }) {mode = "simd"} : () -> ()
    }) : () -> ()
    return
  }

// CHECK-LABEL: func.func @u32_to_u16
// CHECK: ave.hir.vtrunci %{{.*}} {part = #ave.vcvt_part_type<part_even>, uni = #hivm.unsigned_mode<ui2ui>} : vector<64xi32>, vector<128xi16>
  func.func @u32_to_u16(
      %sm: memref<64xui32, #hivm.address_space<ub>>,
      %dm: memref<128xui16, #hivm.address_space<ub>>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %cs = arith.constant 64 : index
    %cd = arith.constant 128 : index
    %s = tla.tensor_desc %sm shape [%c1, %cs, %c1, %c1] stride [%cs, %c1, %c1, %c1] origin_shape [%c1, %cs] coord [%c0, %c0] : memref<64xui32, #hivm.address_space<ub>> -> !s_u32_to_u16
    %d = tla.tensor_desc %dm shape [%c1, %cd, %c1, %c1] stride [%cd, %c1, %c1, %c1] origin_shape [%c1, %cd] coord [%c0, %c0] : memref<128xui16, #hivm.address_space<ub>> -> !d_u32_to_u16
    "tla.vector"() ({
      "tla.vec.func"() ({
        %ss = "tla.make_shape"() : () -> !tla.shape<64>
        %ds = "tla.make_shape"() : () -> !tla.shape<128>
        %co = "tla.make_coord"() : () -> !tla.coord<0>
        %st = "tla.tile_view"(%s, %ss, %co) : (!s_u32_to_u16, !tla.shape<64>, !tla.coord<0>) -> !s_u32_to_u16
        %dt = "tla.tile_view"(%d, %ds, %co) : (!d_u32_to_u16, !tla.shape<128>, !tla.coord<0>) -> !d_u32_to_u16
        %r = "tla.load"(%st) : (!s_u32_to_u16) -> !tla.vector<64xui32>
        // trait = [reg_slot, sat_mode, round_mode]
        %o = "tla.cast"(%r) <{trait = array<i32: 0, 2, 4>}> : (!tla.vector<64xui32>) -> !tla.vector<128xui16>
        "tla.store"(%dt, %o) <{operandSegmentSizes = array<i32: 1, 1, 0, 0, 0>}> : (!d_u32_to_u16, !tla.vector<128xui16>) -> ()
      }) {mode = "simd"} : () -> ()
    }) : () -> ()
    return
  }

// CHECK-LABEL: func.func @u32_to_u8
// CHECK: ave.hir.vtrunci %{{.*}} {pp = #ave.vcvt_pp_type<pp0>, uni = #hivm.unsigned_mode<ui2ui>} : vector<64xi32>, vector<256xi8>
  func.func @u32_to_u8(
      %sm: memref<64xui32, #hivm.address_space<ub>>,
      %dm: memref<256xui8, #hivm.address_space<ub>>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %cs = arith.constant 64 : index
    %cd = arith.constant 256 : index
    %s = tla.tensor_desc %sm shape [%c1, %cs, %c1, %c1] stride [%cs, %c1, %c1, %c1] origin_shape [%c1, %cs] coord [%c0, %c0] : memref<64xui32, #hivm.address_space<ub>> -> !s_u32_to_u8
    %d = tla.tensor_desc %dm shape [%c1, %cd, %c1, %c1] stride [%cd, %c1, %c1, %c1] origin_shape [%c1, %cd] coord [%c0, %c0] : memref<256xui8, #hivm.address_space<ub>> -> !d_u32_to_u8
    "tla.vector"() ({
      "tla.vec.func"() ({
        %ss = "tla.make_shape"() : () -> !tla.shape<64>
        %ds = "tla.make_shape"() : () -> !tla.shape<256>
        %co = "tla.make_coord"() : () -> !tla.coord<0>
        %st = "tla.tile_view"(%s, %ss, %co) : (!s_u32_to_u8, !tla.shape<64>, !tla.coord<0>) -> !s_u32_to_u8
        %dt = "tla.tile_view"(%d, %ds, %co) : (!d_u32_to_u8, !tla.shape<256>, !tla.coord<0>) -> !d_u32_to_u8
        %r = "tla.load"(%st) : (!s_u32_to_u8) -> !tla.vector<64xui32>
        // trait = [reg_slot, sat_mode, round_mode]
        %o = "tla.cast"(%r) <{trait = array<i32: 0, 2, 4>}> : (!tla.vector<64xui32>) -> !tla.vector<256xui8>
        "tla.store"(%dt, %o) <{operandSegmentSizes = array<i32: 1, 1, 0, 0, 0>}> : (!d_u32_to_u8, !tla.vector<256xui8>) -> ()
      }) {mode = "simd"} : () -> ()
    }) : () -> ()
    return
  }

// CHECK-LABEL: func.func @u8_to_f16
// CHECK: ave.hir.vuitofp %{{.*}}, <part_even>, %{{.*}} : vector<256xi8>, vector<256xi1>, vector<128xf16>
// CHECK-NOT: ave.hir.vsitofp
  func.func @u8_to_f16(
      %sm: memref<256xui8, #hivm.address_space<ub>>,
      %dm: memref<128xf16, #hivm.address_space<ub>>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %cs = arith.constant 256 : index
    %cd = arith.constant 128 : index
    %s = tla.tensor_desc %sm shape [%c1, %cs, %c1, %c1] stride [%cs, %c1, %c1, %c1] origin_shape [%c1, %cs] coord [%c0, %c0] : memref<256xui8, #hivm.address_space<ub>> -> !s_u8_to_f16
    %d = tla.tensor_desc %dm shape [%c1, %cd, %c1, %c1] stride [%cd, %c1, %c1, %c1] origin_shape [%c1, %cd] coord [%c0, %c0] : memref<128xf16, #hivm.address_space<ub>> -> !d_u8_to_f16
    "tla.vector"() ({
      "tla.vec.func"() ({
        %ss = "tla.make_shape"() : () -> !tla.shape<256>
        %ds = "tla.make_shape"() : () -> !tla.shape<128>
        %co = "tla.make_coord"() : () -> !tla.coord<0>
        %st = "tla.tile_view"(%s, %ss, %co) : (!s_u8_to_f16, !tla.shape<256>, !tla.coord<0>) -> !s_u8_to_f16
        %dt = "tla.tile_view"(%d, %ds, %co) : (!d_u8_to_f16, !tla.shape<128>, !tla.coord<0>) -> !d_u8_to_f16
        %r = "tla.load"(%st) : (!s_u8_to_f16) -> !tla.vector<256xui8>
        // trait = [reg_slot, sat_mode, round_mode]
        %o = "tla.cast"(%r) <{trait = array<i32: 0, 0, 4>}> : (!tla.vector<256xui8>) -> !tla.vector<128xf16>
        "tla.store"(%dt, %o) <{operandSegmentSizes = array<i32: 1, 1, 0, 0, 0>}> : (!d_u8_to_f16, !tla.vector<128xf16>) -> ()
      }) {mode = "simd"} : () -> ()
    }) : () -> ()
    return
  }

// CHECK-LABEL: func.func @f16_to_u8
// CHECK: ave.hir.vfptoui %{{.*}} : vector<128xf16>, vector<256xi1>, vector<256xi8>
// CHECK-NOT: ave.hir.vfptosi
  func.func @f16_to_u8(
      %sm: memref<128xf16, #hivm.address_space<ub>>,
      %dm: memref<256xui8, #hivm.address_space<ub>>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %cs = arith.constant 128 : index
    %cd = arith.constant 256 : index
    %s = tla.tensor_desc %sm shape [%c1, %cs, %c1, %c1] stride [%cs, %c1, %c1, %c1] origin_shape [%c1, %cs] coord [%c0, %c0] : memref<128xf16, #hivm.address_space<ub>> -> !s_f16_to_u8
    %d = tla.tensor_desc %dm shape [%c1, %cd, %c1, %c1] stride [%cd, %c1, %c1, %c1] origin_shape [%c1, %cd] coord [%c0, %c0] : memref<256xui8, #hivm.address_space<ub>> -> !d_f16_to_u8
    "tla.vector"() ({
      "tla.vec.func"() ({
        %ss = "tla.make_shape"() : () -> !tla.shape<128>
        %ds = "tla.make_shape"() : () -> !tla.shape<256>
        %co = "tla.make_coord"() : () -> !tla.coord<0>
        %st = "tla.tile_view"(%s, %ss, %co) : (!s_f16_to_u8, !tla.shape<128>, !tla.coord<0>) -> !s_f16_to_u8
        %dt = "tla.tile_view"(%d, %ds, %co) : (!d_f16_to_u8, !tla.shape<256>, !tla.coord<0>) -> !d_f16_to_u8
        %r = "tla.load"(%st) : (!s_f16_to_u8) -> !tla.vector<128xf16>
        // trait = [reg_slot, sat_mode, round_mode]
        %o = "tla.cast"(%r) <{trait = array<i32: 0, 2, 0>}> : (!tla.vector<128xf16>) -> !tla.vector<256xui8>
        "tla.store"(%dt, %o) <{operandSegmentSizes = array<i32: 1, 1, 0, 0, 0>}> : (!d_f16_to_u8, !tla.vector<256xui8>) -> ()
      }) {mode = "simd"} : () -> ()
    }) : () -> ()
    return
  }

// CHECK-LABEL: func.func @u8_to_f32
// CHECK: ave.hir.vand
// CHECK: ave.hir.vextui %{{.*}} {pp = #ave.vcvt_pp_type<pp0>} : vector<256xi8>, vector<64xi32>
// CHECK: ave.hir.vsitofp %{{.*}} : vector<64xi32>, vector<256xi1>, vector<64xf32>
// CHECK-NOT: ave.hir.vuitofp
  func.func @u8_to_f32(
      %sm: memref<256xui8, #hivm.address_space<ub>>,
      %dm: memref<64xf32, #hivm.address_space<ub>>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %cs = arith.constant 256 : index
    %cd = arith.constant 64 : index
    %s = tla.tensor_desc %sm shape [%c1, %cs, %c1, %c1] stride [%cs, %c1, %c1, %c1] origin_shape [%c1, %cs] coord [%c0, %c0] : memref<256xui8, #hivm.address_space<ub>> -> !s_u8_to_f32
    %d = tla.tensor_desc %dm shape [%c1, %cd, %c1, %c1] stride [%cd, %c1, %c1, %c1] origin_shape [%c1, %cd] coord [%c0, %c0] : memref<64xf32, #hivm.address_space<ub>> -> !d_u8_to_f32
    "tla.vector"() ({
      "tla.vec.func"() ({
        %ss = "tla.make_shape"() : () -> !tla.shape<256>
        %ds = "tla.make_shape"() : () -> !tla.shape<64>
        %co = "tla.make_coord"() : () -> !tla.coord<0>
        %st = "tla.tile_view"(%s, %ss, %co) : (!s_u8_to_f32, !tla.shape<256>, !tla.coord<0>) -> !s_u8_to_f32
        %dt = "tla.tile_view"(%d, %ds, %co) : (!d_u8_to_f32, !tla.shape<64>, !tla.coord<0>) -> !d_u8_to_f32
        %r = "tla.load"(%st) : (!s_u8_to_f32) -> !tla.vector<256xui8>
        %mask = "tla.create_mask"() {pattern = "M3", dtype = ui8} : () -> !tla.mask<256>
        // trait = [reg_slot, sat_mode, round_mode]
        %o = "tla.cast"(%r, %mask) <{trait = array<i32: 0, 0, 4>}> : (!tla.vector<256xui8>, !tla.mask<256>) -> !tla.vector<64xf32>
        "tla.store"(%dt, %o) <{operandSegmentSizes = array<i32: 1, 1, 0, 0, 0>}> : (!d_u8_to_f32, !tla.vector<64xf32>) -> ()
      }) {mode = "simd"} : () -> ()
    }) : () -> ()
    return
  }

// u16 max(65535, 1) / min(65535, 1) and reduce of the same bound must not
// select the signed twins (vmax/vmin/...): those flip the comparison above
// the signed i16 range.
// CHECK-LABEL: func.func @u16_vec_max_min
// CHECK: ave.hir.vumax
// CHECK: ave.hir.vumin
// CHECK-NOT: ave.hir.vmax %
// CHECK-NOT: ave.hir.vmin %
  func.func @u16_vec_max_min(
      %a: memref<128xui16, #hivm.address_space<ub>>,
      %b: memref<128xui16, #hivm.address_space<ub>>,
      %out: memref<128xui16, #hivm.address_space<ub>>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c128 = arith.constant 128 : index
    %ta = tla.tensor_desc %a shape [%c1, %c128, %c1, %c1] stride [%c128, %c1, %c1, %c1] origin_shape [%c1, %c128] coord [%c0, %c0] : memref<128xui16, #hivm.address_space<ub>> -> !u16_vec
    %tb = tla.tensor_desc %b shape [%c1, %c128, %c1, %c1] stride [%c128, %c1, %c1, %c1] origin_shape [%c1, %c128] coord [%c0, %c0] : memref<128xui16, #hivm.address_space<ub>> -> !u16_vec
    %to = tla.tensor_desc %out shape [%c1, %c128, %c1, %c1] stride [%c128, %c1, %c1, %c1] origin_shape [%c1, %c128] coord [%c0, %c0] : memref<128xui16, #hivm.address_space<ub>> -> !u16_vec
    "tla.vector"() ({
      "tla.vec.func"() ({
        %va = tla.load %ta : !u16_vec -> !tla.vector<128xui16>
        %vb = tla.load %tb : !u16_vec -> !tla.vector<128xui16>
        %mx = tla.max %va, %vb : !tla.vector<128xui16>, !tla.vector<128xui16> -> !tla.vector<128xui16>
        %mn = tla.min %mx, %vb : !tla.vector<128xui16>, !tla.vector<128xui16> -> !tla.vector<128xui16>
        tla.store %to, %mn : !u16_vec, !tla.vector<128xui16>
      }) {mode = "simd"} : () -> ()
    }) : () -> ()
    return
  }

// CHECK-LABEL: func.func @u16_scalar_max_min
// CHECK: ave.hir.vumaxs
// CHECK: ave.hir.vumins
// CHECK-NOT: ave.hir.vmaxs %
// CHECK-NOT: ave.hir.vmins %
  func.func @u16_scalar_max_min(
      %a: memref<128xui16, #hivm.address_space<ub>>,
      %out: memref<128xui16, #hivm.address_space<ub>>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c128 = arith.constant 128 : index
    %cst = arith.constant 1 : i16
    %ta = tla.tensor_desc %a shape [%c1, %c128, %c1, %c1] stride [%c128, %c1, %c1, %c1] origin_shape [%c1, %c128] coord [%c0, %c0] : memref<128xui16, #hivm.address_space<ub>> -> !u16_vec
    %to = tla.tensor_desc %out shape [%c1, %c128, %c1, %c1] stride [%c128, %c1, %c1, %c1] origin_shape [%c1, %c128] coord [%c0, %c0] : memref<128xui16, #hivm.address_space<ub>> -> !u16_vec
    "tla.vector"() ({
      "tla.vec.func"() ({
        %va = tla.load %ta : !u16_vec -> !tla.vector<128xui16>
        %mx = tla.maxs %va, %cst : !tla.vector<128xui16>, i16 -> !tla.vector<128xui16>
        %mn = tla.mins %mx, %cst : !tla.vector<128xui16>, i16 -> !tla.vector<128xui16>
        tla.store %to, %mn : !u16_vec, !tla.vector<128xui16>
      }) {mode = "simd"} : () -> ()
    }) : () -> ()
    return
  }

// CHECK-LABEL: func.func @u16_reduce_max_min
// CHECK: ave.hir.reduction <umax>
// CHECK: ave.hir.reduction <umin>
// CHECK-NOT: ave.hir.reduction <max>
// CHECK-NOT: ave.hir.reduction <min>
  func.func @u16_reduce_max_min(
      %a: memref<128xui16, #hivm.address_space<ub>>,
      %out_max: memref<1xui16, #hivm.address_space<ub>>,
      %out_min: memref<1xui16, #hivm.address_space<ub>>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c128 = arith.constant 128 : index
    %ta = tla.tensor_desc %a shape [%c1, %c128, %c1, %c1] stride [%c128, %c1, %c1, %c1] origin_shape [%c1, %c128] coord [%c0, %c0] : memref<128xui16, #hivm.address_space<ub>> -> !u16_vec
    %tmax = tla.tensor_desc %out_max shape [%c1, %c1, %c1, %c1] stride [%c1, %c1, %c1, %c1] origin_shape [%c1, %c1] coord [%c0, %c0] : memref<1xui16, #hivm.address_space<ub>> -> !u16_vec1
    %tmin = tla.tensor_desc %out_min shape [%c1, %c1, %c1, %c1] stride [%c1, %c1, %c1, %c1] origin_shape [%c1, %c1] coord [%c0, %c0] : memref<1xui16, #hivm.address_space<ub>> -> !u16_vec1
    "tla.vector"() ({
      "tla.vec.func"() ({
        %va = tla.load %ta : !u16_vec -> !tla.vector<128xui16>
        %mask = "tla.create_mask"() {pattern = "ALL", dtype = ui16} : () -> !tla.mask<128>
        %rmax = tla.reduce %va mask %mask {kind = "max"} : !tla.vector<128xui16> mask !tla.mask<128> -> !tla.vector<1xui16>
        %rmin = tla.reduce %va mask %mask {kind = "min"} : !tla.vector<128xui16> mask !tla.mask<128> -> !tla.vector<1xui16>
        tla.store %tmax, %rmax : !u16_vec1, !tla.vector<1xui16>
        tla.store %tmin, %rmin : !u16_vec1, !tla.vector<1xui16>
      }) {mode = "simd"} : () -> ()
    }) : () -> ()
    return
  }
}
