// RUN: %tla_compile %s -o - | %filecheck %s --implicit-check-not=_mlir_ciface_cast_
// RUN: sed 's/\[1, 1, 7\]/[1, 1, 4]/' %s | %tla_compile - -o - | %filecheck %s --check-prefix=HIF8-DEFAULT

!bf16_reg = !tla.tensor<!tla.layout<!tla.shape<128>, !tla.stride<1>, !tla.shape<128>, RowMajor>, !tla.coord<0>, !tla.ptr<bf16, ub, 32>>
!f16_reg = !tla.tensor<!tla.layout<!tla.shape<128>, !tla.stride<1>, !tla.shape<128>, RowMajor>, !tla.coord<0>, !tla.ptr<f16, ub, 32>>
!f32_reg = !tla.tensor<!tla.layout<!tla.shape<64>, !tla.stride<1>, !tla.shape<64>, RowMajor>, !tla.coord<0>, !tla.ptr<f32, ub, 32>>
!i32_reg = !tla.tensor<!tla.layout<!tla.shape<64>, !tla.stride<1>, !tla.shape<64>, RowMajor>, !tla.coord<0>, !tla.ptr<i32, ub, 32>>
!i64_reg = !tla.tensor<!tla.layout<!tla.shape<32>, !tla.stride<1>, !tla.shape<32>, RowMajor>, !tla.coord<0>, !tla.ptr<i64, ub, 32>>

!s16 = !tla.tensor<!tla.layout<!tla.shape<128>, !tla.stride<1>, !tla.shape<128>, RowMajor>, !tla.coord<0>, !tla.ptr<i16, ub, 32>>
!s32 = !tla.tensor<!tla.layout<!tla.shape<64>, !tla.stride<1>, !tla.shape<64>, RowMajor>, !tla.coord<0>, !tla.ptr<i32, ub, 32>>
!u8 = !tla.tensor<!tla.layout<!tla.shape<256>, !tla.stride<1>, !tla.shape<256>, RowMajor>, !tla.coord<0>, !tla.ptr<ui8, ub, 32>>
!u16 = !tla.tensor<!tla.layout<!tla.shape<128>, !tla.stride<1>, !tla.shape<128>, RowMajor>, !tla.coord<0>, !tla.ptr<ui16, ub, 32>>
!u32 = !tla.tensor<!tla.layout<!tla.shape<64>, !tla.stride<1>, !tla.shape<64>, RowMajor>, !tla.coord<0>, !tla.ptr<ui32, ub, 32>>
!f16 = !tla.tensor<!tla.layout<!tla.shape<128>, !tla.stride<1>, !tla.shape<128>, RowMajor>, !tla.coord<0>, !tla.ptr<f16, ub, 32>>
!i4 = !tla.tensor<!tla.layout<!tla.shape<256>, !tla.stride<1>, !tla.shape<256>, RowMajor>, !tla.coord<0>, !tla.ptr<!tla.i4x2, ub, 32>>
!hif8 = !tla.tensor<!tla.layout<!tla.shape<256>, !tla.stride<1>, !tla.shape<256>, RowMajor>, !tla.coord<0>, !tla.ptr<!tla.hif8, ub, 32>>
!e8m0 = !tla.tensor<!tla.layout<!tla.shape<256>, !tla.stride<1>, !tla.shape<256>, RowMajor>, !tla.coord<0>, !tla.ptr<!tla.f8e8m0, ub, 32>>
!bf16 = !tla.tensor<!tla.layout<!tla.shape<128>, !tla.stride<1>, !tla.shape<128>, RowMajor>, !tla.coord<0>, !tla.ptr<bf16, ub, 32>>

// CHECK-LABEL: func.func @register_casts(
// CHECK-LABEL: func.func private @vector_region_
// CHECK-DAG: %[[CEIL:[0-9]+]] = llvm.mlir.constant(3 : i32)
// CHECK-DAG: %[[ONE:[0-9]+]] = llvm.mlir.constant(1 : i32)
// CHECK-DAG: %[[FLOOR:[0-9]+]] = llvm.mlir.constant(2 : i32)
// CHECK-DAG: %[[ZERO:[0-9]+]] = llvm.mlir.constant(0 : i32)
// CHECK: %[[MASK64:[0-9]+]] = "hivm_regbaseintrins.intr.hivm.punpack"(%{{[0-9]+}}, %[[ZERO]])
// CHECK: "hivm_regbaseintrins.intr.hivm.vcvtff.bf162f16.x"(%{{[0-9]+}}, %{{[0-9]+}}, %[[ONE]], %[[FLOOR]]) : (vector<128xbf16>, vector<256xi1>, i32, i32) -> vector<128xf16>
// CHECK: %[[TRUNC:[0-9]+]] = llvm.mlir.constant(4 : i32)
// CHECK: %[[ODD_PART:[0-9]+]] = llvm.mlir.constant(1 : i32)
// CHECK: "hivm_regbaseintrins.intr.hivm.vcvtfi.f162s32.x"(%{{[0-9]+}}, %{{[0-9]+}}, %[[TRUNC]], %[[ODD_PART]]) : (vector<128xf16>, vector<256xi1>, i32, i32) -> vector<64xi32>
// CHECK: "hivm_regbaseintrins.intr.hivm.vcvtfi.f322s64.x"(%{{[0-9]+}}, %{{[0-9]+}}, %[[CEIL]], %[[ONE]], %[[ONE]]) : (vector<64xf32>, vector<256xi1>, i32, i32, i32) -> vector<32xi64>
// CHECK: "hivm_regbaseintrins.intr.hivm.vcvtif.s642f32.x"(%{{[0-9]+}}, %[[MASK64]], %[[ONE]], %[[ZERO]]) : (vector<32xi64>, vector<256xi1>, i32, i32) -> vector<64xf32>
// CHECK: "hivm_regbaseintrins.intr.hivm.vcvtii.s322s64.x"(%{{[0-9]+}}, %{{[0-9]+}}, %[[ZERO]]) : (vector<64xi32>, vector<256xi1>, i32) -> vector<32xi64>
// CHECK: "hivm_regbaseintrins.intr.hivm.vcvtii.s642s32.x"(%{{[0-9]+}}, %[[MASK64]], %[[ONE]], %[[ONE]]) : (vector<32xi64>, vector<256xi1>, i32, i32) -> vector<64xi32>

module {
  func.func @register_casts(
      %bf16: memref<128xbf16, #hivm.address_space<ub>>,
      %f16: memref<128xf16, #hivm.address_space<ub>>,
      %f32: memref<64xf32, #hivm.address_space<ub>>,
      %i32: memref<64xi32, #hivm.address_space<ub>>,
      %i64: memref<32xi64, #hivm.address_space<ub>>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c32 = arith.constant 32 : index
    %c64 = arith.constant 64 : index
    %c128 = arith.constant 128 : index
    %tb = tla.tensor_desc %bf16 shape [%c1, %c128, %c1, %c1] stride [%c128, %c1, %c1, %c1] origin_shape [%c1, %c128] coord [%c0, %c0] : memref<128xbf16, #hivm.address_space<ub>> -> !bf16_reg
    %th = tla.tensor_desc %f16 shape [%c1, %c128, %c1, %c1] stride [%c128, %c1, %c1, %c1] origin_shape [%c1, %c128] coord [%c0, %c0] : memref<128xf16, #hivm.address_space<ub>> -> !f16_reg
    %tf = tla.tensor_desc %f32 shape [%c1, %c64, %c1, %c1] stride [%c64, %c1, %c1, %c1] origin_shape [%c1, %c64] coord [%c0, %c0] : memref<64xf32, #hivm.address_space<ub>> -> !f32_reg
    %ti = tla.tensor_desc %i32 shape [%c1, %c64, %c1, %c1] stride [%c64, %c1, %c1, %c1] origin_shape [%c1, %c64] coord [%c0, %c0] : memref<64xi32, #hivm.address_space<ub>> -> !i32_reg
    %tl = tla.tensor_desc %i64 shape [%c1, %c32, %c1, %c1] stride [%c32, %c1, %c1, %c1] origin_shape [%c1, %c32] coord [%c0, %c0] : memref<32xi64, #hivm.address_space<ub>> -> !i64_reg
    "tla.vec.func"() ({
      %b = tla.load %tb : !bf16_reg -> !tla.vector<128xbf16>
      %h = tla.load %th : !f16_reg -> !tla.vector<128xf16>
      %f = tla.load %tf : !f32_reg -> !tla.vector<64xf32>
      %i = tla.load %ti : !i32_reg -> !tla.vector<64xi32>
      %l = tla.load %tl : !i64_reg -> !tla.vector<32xi64>
      %mask16 = "tla.create_mask"() {pattern = "M3", dtype = bf16} : () -> !tla.mask<128>
      %mask64 = "tla.create_mask"() {pattern = "VL3", dtype = i64} : () -> !tla.mask<32>
      // Traits are [register slot, saturation mode, rounding mode].
      %bh = "tla.cast"(%b, %mask16) <{trait = array<i32: 0, 1, 1>}> : (!tla.vector<128xbf16>, !tla.mask<128>) -> !tla.vector<128xf16>
      %hi = "tla.cast"(%h) <{trait = array<i32: 1, 0, 3>}> : (!tla.vector<128xf16>) -> !tla.vector<64xi32>
      %fl = "tla.cast"(%f) <{trait = array<i32: 1, 1, 2>}> : (!tla.vector<64xf32>) -> !tla.vector<32xi64>
      %lf = "tla.cast"(%l, %mask64) <{trait = array<i32: 0, 0, 0>}> : (!tla.vector<32xi64>, !tla.mask<32>) -> !tla.vector<?xf32>
      %il = "tla.cast"(%i) <{trait = array<i32: 0, 0, 4>}> : (!tla.vector<64xi32>) -> !tla.vector<32xi64>
      %li = "tla.cast"(%l, %mask64) <{trait = array<i32: 1, 1, 4>}> : (!tla.vector<32xi64>, !tla.mask<32>) -> !tla.vector<?xi32>
      tla.store %th, %bh : !f16_reg, !tla.vector<128xf16>
      tla.store %ti, %hi : !i32_reg, !tla.vector<64xi32>
      tla.store %tl, %fl : !i64_reg, !tla.vector<32xi64>
      tla.store %tf, %lf : !f32_reg, !tla.vector<?xf32>
      tla.store %tl, %il : !i64_reg, !tla.vector<32xi64>
      tla.store %ti, %li : !i32_reg, !tla.vector<?xi32>
    }) {mode = "simd"} : () -> ()
    return
  }

  // CHECK-LABEL: func.func @additional_casts(
  // CHECK-LABEL: func.func private @vector_region_
  // CHECK: %[[SLOT3:[0-9]+]] = llvm.mlir.constant(3 : i32)
  // CHECK: %[[NOSAT:[0-9]+]] = llvm.mlir.constant(0 : i32)
  // CHECK: %[[ONE:[0-9]+]] = llvm.mlir.constant(1 : i32)
  // CHECK-DAG: %[[ALL8:[0-9]+]] = "hivm_regbaseintrins.intr.hivm.pge.b8"
  // CHECK-DAG: %[[ALL16:[0-9]+]] = "hivm_regbaseintrins.intr.hivm.pge.b16"
  // CHECK-DAG: %[[ALL32:[0-9]+]] = "hivm_regbaseintrins.intr.hivm.pge.b32"
  // CHECK: %[[S16:[0-9]+]] = "hivm_regbaseintrins.intr.hivm.vldsx1.v128s16"
  // CHECK: %[[S32:[0-9]+]] = "hivm_regbaseintrins.intr.hivm.vldsx1.v64s32"
  // CHECK: %[[U32:[0-9]+]] = "hivm_regbaseintrins.intr.hivm.vldsx1.v64u32"
  // CHECK: %[[F16:[0-9]+]] = "hivm_regbaseintrins.intr.hivm.vldsx1.v128f16"
  // CHECK: %[[HIF8:[0-9]+]] = "hivm_regbaseintrins.intr.hivm.vldsx1.v256s8"
  // CHECK: %[[E8M0:[0-9]+]] = "hivm_regbaseintrins.intr.hivm.vldsx1.v256s8"
  // CHECK: %[[VL3:[0-9]+]] = llvm.mlir.constant(3 : i32)
  // CHECK: %[[PAIR_MASK:[0-9]+]] = "hivm_regbaseintrins.intr.hivm.pge.b32"(%[[VL3]], %{{[0-9]+}})
  // CHECK: %[[M3_8:[0-9]+]] = llvm.mlir.constant(10 : i32)
  // CHECK: %[[SPARSE8:[0-9]+]] = "hivm_regbaseintrins.intr.hivm.pge.b8"(%[[M3_8]], %{{[0-9]+}})
  // CHECK: %[[M3_16:[0-9]+]] = llvm.mlir.constant(10 : i32)
  // CHECK: %[[SPARSE16:[0-9]+]] = "hivm_regbaseintrins.intr.hivm.pge.b16"(%[[M3_16]], %{{[0-9]+}})
  // CHECK: "hivm_regbaseintrins.intr.hivm.vcvtii.s162u8.x"(%[[S16]], %[[SPARSE16]], %[[ONE]], %[[ONE]]) : (vector<128xi16>, vector<256xi1>, i32, i32) -> vector<256xi8>
  // CHECK: "hivm_regbaseintrins.intr.hivm.vcvtii.s162u32.x"(%[[S16]], %[[ALL16]], %[[ONE]]) : (vector<128xi16>, vector<256xi1>, i32) -> vector<64xi32>
  // CHECK: "hivm_regbaseintrins.intr.hivm.vcvtii.s322u8.x"(%[[S32]], %[[ALL32]], %[[NOSAT]], %[[SLOT3]]) : (vector<64xi32>, vector<256xi1>, i32, i32) -> vector<256xi8>
  // CHECK: "hivm_regbaseintrins.intr.hivm.vcvtii.s322u16.x"(%[[S32]], %[[ALL32]], %[[ONE]], %[[ONE]]) : (vector<64xi32>, vector<256xi1>, i32, i32) -> vector<128xi16>
  // CHECK: "hivm_regbaseintrins.intr.hivm.vcvtii.u322s16.x"(%[[U32]], %[[ALL32]], %[[ONE]], %[[ONE]]) : (vector<64xi32>, vector<256xi1>, i32, i32) -> vector<128xi16>
  // CHECK: call @lowp_cast_i16_to_i4x2_sat_slot3(%[[S16]], %[[ALL32]]) : (vector<128xi16>, vector<256xi1>) -> vector<256xi8>
  // CHECK: call @lowp_cast_f16_to_i4x2_floor_nosat_slot2(%[[F16]], %[[PAIR_MASK]]) : (vector<128xf16>, vector<256xi1>) -> vector<256xi8>
  // CHECK: call @lowp_cast_f16_to_hif8_hybrid_sat_slot1(%[[F16]], %[[ALL16]]) : (vector<128xf16>, vector<256xi1>) -> vector<256xi8>
  // HIF8-DEFAULT: call @lowp_cast_f16_to_hif8_round_sat_slot1
  // CHECK: call @lowp_cast_hif8_to_f16_slot1(%[[HIF8]], %[[ALL8]]) : (vector<256xi8>, vector<256xi1>) -> vector<128xf16>
  // CHECK: call @lowp_cast_e8m0_to_bf16_slot1(%[[E8M0]], %[[SPARSE8]]) : (vector<256xi8>, vector<256xi1>) -> vector<128xbf16>
  func.func @additional_casts(
      %s16: memref<128xi16, #hivm.address_space<ub>>,
      %s32: memref<64xi32, #hivm.address_space<ub>>,
      %u8: memref<256xui8, #hivm.address_space<ub>>,
      %u16: memref<128xui16, #hivm.address_space<ub>>,
      %u32: memref<64xui32, #hivm.address_space<ub>>,
      %f16: memref<128xf16, #hivm.address_space<ub>>,
      %i4: memref<256xi8, #hivm.address_space<ub>>,
      %hif8: memref<256xi8, #hivm.address_space<ub>>,
      %e8m0: memref<256xi8, #hivm.address_space<ub>>,
      %bf16: memref<128xbf16, #hivm.address_space<ub>>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c64 = arith.constant 64 : index
    %c128 = arith.constant 128 : index
    %c256 = arith.constant 256 : index
    %t_s16 = tla.tensor_desc %s16 shape [%c1, %c128, %c1, %c1] stride [%c128, %c1, %c1, %c1] origin_shape [%c1, %c128] coord [%c0, %c0] : memref<128xi16, #hivm.address_space<ub>> -> !s16
    %t_s32 = tla.tensor_desc %s32 shape [%c1, %c64, %c1, %c1] stride [%c64, %c1, %c1, %c1] origin_shape [%c1, %c64] coord [%c0, %c0] : memref<64xi32, #hivm.address_space<ub>> -> !s32
    %t_u8 = tla.tensor_desc %u8 shape [%c1, %c256, %c1, %c1] stride [%c256, %c1, %c1, %c1] origin_shape [%c1, %c256] coord [%c0, %c0] : memref<256xui8, #hivm.address_space<ub>> -> !u8
    %t_u16 = tla.tensor_desc %u16 shape [%c1, %c128, %c1, %c1] stride [%c128, %c1, %c1, %c1] origin_shape [%c1, %c128] coord [%c0, %c0] : memref<128xui16, #hivm.address_space<ub>> -> !u16
    %t_u32 = tla.tensor_desc %u32 shape [%c1, %c64, %c1, %c1] stride [%c64, %c1, %c1, %c1] origin_shape [%c1, %c64] coord [%c0, %c0] : memref<64xui32, #hivm.address_space<ub>> -> !u32
    %t_f16 = tla.tensor_desc %f16 shape [%c1, %c128, %c1, %c1] stride [%c128, %c1, %c1, %c1] origin_shape [%c1, %c128] coord [%c0, %c0] : memref<128xf16, #hivm.address_space<ub>> -> !f16
    %t_i4 = tla.tensor_desc %i4 shape [%c1, %c256, %c1, %c1] stride [%c256, %c1, %c1, %c1] origin_shape [%c1, %c256] coord [%c0, %c0] : memref<256xi8, #hivm.address_space<ub>> -> !i4
    %t_hif8 = tla.tensor_desc %hif8 shape [%c1, %c256, %c1, %c1] stride [%c256, %c1, %c1, %c1] origin_shape [%c1, %c256] coord [%c0, %c0] : memref<256xi8, #hivm.address_space<ub>> -> !hif8
    %t_e8m0 = tla.tensor_desc %e8m0 shape [%c1, %c256, %c1, %c1] stride [%c256, %c1, %c1, %c1] origin_shape [%c1, %c256] coord [%c0, %c0] : memref<256xi8, #hivm.address_space<ub>> -> !e8m0
    %t_bf16 = tla.tensor_desc %bf16 shape [%c1, %c128, %c1, %c1] stride [%c128, %c1, %c1, %c1] origin_shape [%c1, %c128] coord [%c0, %c0] : memref<128xbf16, #hivm.address_space<ub>> -> !bf16
    "tla.vec.func"() ({
      %v_s16 = tla.load %t_s16 : !s16 -> !tla.vector<128xi16>
      %v_s32 = tla.load %t_s32 : !s32 -> !tla.vector<64xi32>
      %v_u32 = tla.load %t_u32 : !u32 -> !tla.vector<64xui32>
      %v_f16 = tla.load %t_f16 : !f16 -> !tla.vector<128xf16>
      %v_hif8 = tla.load %t_hif8 : !hif8 -> !tla.vector<256x!tla.hif8>
      %v_e8m0 = tla.load %t_e8m0 : !e8m0 -> !tla.vector<256x!tla.f8e8m0>
      %p32 = "tla.create_mask"() {pattern = "VL3", dtype = ui32} : () -> !tla.mask<64>
      %p8 = "tla.create_mask"() {pattern = "M3", dtype = ui8} : () -> !tla.mask<256>
      %p16 = "tla.create_mask"() {pattern = "M3", dtype = i16} : () -> !tla.mask<128>
      %r0 = tla.cast %v_s16, [1, 1, 4] mask %p16 : !tla.vector<128xi16> mask !tla.mask<128> -> !tla.vector<?xui8>
      %r1 = tla.cast %v_s16, [1, 0, 4] : !tla.vector<128xi16> -> !tla.vector<64xui32>
      %r2 = tla.cast %v_s32, [3, 2, 4] : !tla.vector<64xi32> -> !tla.vector<?xui8>
      %r3 = tla.cast %v_s32, [1, 1, 4] : !tla.vector<64xi32> -> !tla.vector<?xui16>
      %r4 = tla.cast %v_u32, [1, 1, 4] : !tla.vector<64xui32> -> !tla.vector<?xi16>
      %r5 = tla.cast %v_s16, [3, 1, 4] : !tla.vector<128xi16> -> !tla.vector<?x!tla.i4x2>
      %r6 = tla.cast %v_f16, [2, 2, 1] mask %p32 : !tla.vector<128xf16> mask !tla.mask<64> -> !tla.vector<?x!tla.i4x2>
      %r7 = tla.cast %v_f16, [1, 1, 7] : !tla.vector<128xf16> -> !tla.vector<?x!tla.hif8>
      %r8 = tla.cast %v_hif8, [1, 0, 4] : !tla.vector<256x!tla.hif8> -> !tla.vector<128xf16>
      %r9 = tla.cast %v_e8m0, [1, 0, 4] mask %p8 : !tla.vector<256x!tla.f8e8m0> mask !tla.mask<256> -> !tla.vector<128xbf16>
      tla.store %t_u8, %r0 : !u8, !tla.vector<?xui8>
      tla.store %t_u32, %r1 : !u32, !tla.vector<64xui32>
      tla.store %t_u8, %r2 : !u8, !tla.vector<?xui8>
      tla.store %t_u16, %r3 : !u16, !tla.vector<?xui16>
      tla.store %t_s16, %r4 : !s16, !tla.vector<?xi16>
      tla.store %t_i4, %r5 : !i4, !tla.vector<?x!tla.i4x2>
      tla.store %t_i4, %r6 : !i4, !tla.vector<?x!tla.i4x2>
      tla.store %t_hif8, %r7 : !hif8, !tla.vector<?x!tla.hif8>
      tla.store %t_f16, %r8 : !f16, !tla.vector<128xf16>
      tla.store %t_bf16, %r9 : !bf16, !tla.vector<128xbf16>
    }) {mode = "simd"} : () -> ()
    return
  }
}
