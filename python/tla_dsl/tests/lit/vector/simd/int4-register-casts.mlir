// RUN: %tla_compile %s -o - | %filecheck %s
//
// i4 (int4x2_t) widens through the dedicated regbase intrinsics. The pp
// operand is the register slot; the IR-level cast is always CAST_NONE.

!s_i4 = !tla.tensor<!tla.layout<!tla.shape<256>, !tla.stride<1>, !tla.shape<256>, RowMajor>, !tla.coord<0>, !tla.ptr<!tla.i4x2, ub, 32>>
!d_f16 = !tla.tensor<!tla.layout<!tla.shape<128>, !tla.stride<1>, !tla.shape<128>, RowMajor>, !tla.coord<0>, !tla.ptr<f16, ub, 32>>
!d_bf16 = !tla.tensor<!tla.layout<!tla.shape<128>, !tla.stride<1>, !tla.shape<128>, RowMajor>, !tla.coord<0>, !tla.ptr<bf16, ub, 32>>
!d_i16 = !tla.tensor<!tla.layout<!tla.shape<128>, !tla.stride<1>, !tla.shape<128>, RowMajor>, !tla.coord<0>, !tla.ptr<i16, ub, 32>>

module {
  func.func @i4_to_f16_slot3(%sm: memref<256xi8, #hivm.address_space<ub>>, %dm: memref<128xf16, #hivm.address_space<ub>>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %cs = arith.constant 256 : index
    %cd = arith.constant 128 : index
    %s = tla.tensor_desc %sm shape [%c1, %cs, %c1, %c1] stride [%cs, %c1, %c1, %c1] origin_shape [%c1, %cs] coord [%c0, %c0] : memref<256xi8, #hivm.address_space<ub>> -> !s_i4
    %d = tla.tensor_desc %dm shape [%c1, %cd, %c1, %c1] stride [%cd, %c1, %c1, %c1] origin_shape [%c1, %cd] coord [%c0, %c0] : memref<128xf16, #hivm.address_space<ub>> -> !d_f16
    "tla.vec.func"() ({
      %v = tla.load %s : !s_i4 -> !tla.vector<256x!tla.i4x2>
      %r = "tla.cast"(%v) <{trait = array<i32: 3, 2, 4>}> : (!tla.vector<256x!tla.i4x2>) -> !tla.vector<128xf16>
      tla.store %d, %r : !d_f16, !tla.vector<128xf16>
    }) {mode = "simd"} : () -> ()
    return
  }

  func.func @i4_to_bf16_slot1(%sm: memref<256xi8, #hivm.address_space<ub>>, %dm: memref<128xbf16, #hivm.address_space<ub>>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %cs = arith.constant 256 : index
    %cd = arith.constant 128 : index
    %s = tla.tensor_desc %sm shape [%c1, %cs, %c1, %c1] stride [%cs, %c1, %c1, %c1] origin_shape [%c1, %cs] coord [%c0, %c0] : memref<256xi8, #hivm.address_space<ub>> -> !s_i4
    %d = tla.tensor_desc %dm shape [%c1, %cd, %c1, %c1] stride [%cd, %c1, %c1, %c1] origin_shape [%c1, %cd] coord [%c0, %c0] : memref<128xbf16, #hivm.address_space<ub>> -> !d_bf16
    "tla.vec.func"() ({
      %v = tla.load %s : !s_i4 -> !tla.vector<256x!tla.i4x2>
      %r = "tla.cast"(%v) <{trait = array<i32: 1, 2, 4>}> : (!tla.vector<256x!tla.i4x2>) -> !tla.vector<128xbf16>
      tla.store %d, %r : !d_bf16, !tla.vector<128xbf16>
    }) {mode = "simd"} : () -> ()
    return
  }

  func.func @i4_to_i16_slot2(%sm: memref<256xi8, #hivm.address_space<ub>>, %dm: memref<128xi16, #hivm.address_space<ub>>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %cs = arith.constant 256 : index
    %cd = arith.constant 128 : index
    %s = tla.tensor_desc %sm shape [%c1, %cs, %c1, %c1] stride [%cs, %c1, %c1, %c1] origin_shape [%c1, %cs] coord [%c0, %c0] : memref<256xi8, #hivm.address_space<ub>> -> !s_i4
    %d = tla.tensor_desc %dm shape [%c1, %cd, %c1, %c1] stride [%cd, %c1, %c1, %c1] origin_shape [%c1, %cd] coord [%c0, %c0] : memref<128xi16, #hivm.address_space<ub>> -> !d_i16
    "tla.vec.func"() ({
      %v = tla.load %s : !s_i4 -> !tla.vector<256x!tla.i4x2>
      %r = "tla.cast"(%v) <{trait = array<i32: 2, 2, 4>}> : (!tla.vector<256x!tla.i4x2>) -> !tla.vector<128xi16>
      tla.store %d, %r : !d_i16, !tla.vector<128xi16>
    }) {mode = "simd"} : () -> ()
    return
  }
}

// CHECK-LABEL: func.func private @vector_region_0
// CHECK: llvm.mlir.constant(3 : i32)
// CHECK: "hivm_regbaseintrins.intr.hivm.vcvtif.s4x22f16.x"
// CHECK-SAME: (vector<256xi8>, vector<256xi1>, i32) -> vector<128xf16>

// CHECK-LABEL: func.func private @vector_region_1
// CHECK: llvm.mlir.constant(1 : i32)
// CHECK: "hivm_regbaseintrins.intr.hivm.vcvtif.s4x22bf16.x"
// CHECK-SAME: (vector<256xi8>, vector<256xi1>, i32) -> vector<128xbf16>

// CHECK-LABEL: func.func private @vector_region_2
// CHECK: llvm.mlir.constant(2 : i32)
// CHECK: "hivm_regbaseintrins.intr.hivm.vcvtii.s4x22s16.x"
// CHECK-SAME: (vector<256xi8>, vector<256xi1>, i32) -> vector<128xi16>
