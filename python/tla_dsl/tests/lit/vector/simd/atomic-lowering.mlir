// RUN: %tla_compile %s --emit=mlir --mlir-print-ir-after=tla-lower-block-idx -o /dev/null 2>&1 | %filecheck %s --check-prefix=LOWER
// RUN: %tla_compile %s --emit=mlir --mlir-print-ir-after=tla-vector-region-simd -o /dev/null 2>&1 | %filecheck %s --check-prefix=BEFORE
// RUN: %tla_compile %s --emit=mlir -o - | %filecheck %s --check-prefix=FINAL

// A SIMD atomic is an architecture op: tla-lower-ptr only turns its !tla.ptr
// operand into an i64 byte address, and the arch-op lowering pass
// (tla-lower-block-idx) rewrites it into llvm.inttoptr + one atom intrinsic.
// The old value stays in the pointee type so a u32/u64 result can be written
// straight back.

// LOWER-LABEL: func.func @atomic_add_i32
// LOWER: llvm.call_intrinsic "llvm.hivm.atom.ADD.G.s32"({{.*}}) : (!llvm.ptr<1>, i32, i32) -> i32

// LOWER-LABEL: func.func @atomic_add_u32
// The frontend's unsigned literal bridge is peeled: the intrinsic takes the
// signless i32 carrier, and the i32 result is bridged back to ui32.
// LOWER: llvm.call_intrinsic "llvm.hivm.atom.ADD.G.s32"({{.*}}) : (!llvm.ptr<1>, i32, i32) -> i32
// LOWER: builtin.unrealized_conversion_cast {{.*}} : i32 to ui32

// LOWER-LABEL: func.func @atomic_add_f32
// LOWER: llvm.call_intrinsic "llvm.hivm.atom.ADD.G.fp32"({{.*}}) : (!llvm.ptr<1>, f32, i32) -> f32

// LOWER-LABEL: func.func @atomic_add_u64_offset
// A u64 pointee scales the +2 element offset to 16 bytes before the intrinsic.
// LOWER: arith.constant 16 : i64
// LOWER: arith.addi
// LOWER: llvm.call_intrinsic "llvm.hivm.atom.ADD.G.s64"({{.*}}) : (!llvm.ptr<1>, i64, i32) -> i64

// LOWER-LABEL: func.func @atomic_exch_u32
// LOWER: llvm.call_intrinsic "llvm.hivm.atom.EXCH.G.u32"({{.*}}) : (!llvm.ptr<1>, i32, i32) -> i32

// LOWER-LABEL: func.func @atomic_exch_u64
// LOWER: llvm.call_intrinsic "llvm.hivm.atom.EXCH.G.u64"({{.*}}) : (!llvm.ptr<1>, i64, i32) -> i64

// tla.cube: the region is flattened by tla-cube-region and the AIC function
// keeps the atom; the arch-op pass lowers it exactly as it does on AIV.
// LOWER-LABEL: func.func @atomic_add_u32_cube
// LOWER-SAME: hivm.func_core_type = #hivm.func_core_type<AIC>
// LOWER: llvm.call_intrinsic "llvm.hivm.atom.ADD.G.s32"({{.*}}) : (!llvm.ptr<1>, i32, i32) -> i32

// LOWER-NOT: tla.simd_atomic

// The SIMD region pass does not lower the atom; it stays a tla op until the
// arch-op pass consumes it.
// BEFORE-LABEL: func.func @atomic_add_u32
// BEFORE: tla.simd_atomic_add
// BEFORE-NOT: llvm.call_intrinsic

// FINAL-NOT: tla.simd_atomic
// FINAL-NOT: "tla.vector"

!gm_i32 = !tla.tensor<!tla.layout<!tla.shape<8>, !tla.stride<1>, !tla.shape<8>, RowMajor>, !tla.coord<0>, !tla.ptr<i32, gm, 4>>
!gm_u32 = !tla.tensor<!tla.layout<!tla.shape<8>, !tla.stride<1>, !tla.shape<8>, RowMajor>, !tla.coord<0>, !tla.ptr<ui32, gm, 4>>
!gm_f32 = !tla.tensor<!tla.layout<!tla.shape<8>, !tla.stride<1>, !tla.shape<8>, RowMajor>, !tla.coord<0>, !tla.ptr<f32, gm, 4>>
!gm_u64 = !tla.tensor<!tla.layout<!tla.shape<8>, !tla.stride<1>, !tla.shape<8>, RowMajor>, !tla.coord<0>, !tla.ptr<ui64, gm, 8>>

module {
  func.func @atomic_add_i32(%dst_memref: memref<8xi32, #hivm.address_space<gm>>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c8 = arith.constant 8 : index
    %dst = tla.tensor_desc %dst_memref shape [%c1, %c8, %c1, %c1] stride [%c8, %c1, %c1, %c1] origin_shape [%c1, %c8] coord [%c0, %c0] : memref<8xi32, #hivm.address_space<gm>> -> !gm_i32
    "tla.vector"() ({
      %p = tla.tensor_ptr %dst : !gm_i32 -> !tla.ptr<i32, gm, 4>
      %one = arith.constant 1 : i32
      %old = tla.simd_atomic_add %p, %one : !tla.ptr<i32, gm, 4>, i32 -> i32
      tla.scalar_store %dst[%c0], %old : !gm_i32, i32
    }) : () -> ()
    return
  }

  func.func @atomic_add_u32(%dst_memref: memref<8xui32, #hivm.address_space<gm>>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c8 = arith.constant 8 : index
    %dst = tla.tensor_desc %dst_memref shape [%c1, %c8, %c1, %c1] stride [%c8, %c1, %c1, %c1] origin_shape [%c1, %c8] coord [%c0, %c0] : memref<8xui32, #hivm.address_space<gm>> -> !gm_u32
    "tla.vector"() ({
      %p = tla.tensor_ptr %dst : !gm_u32 -> !tla.ptr<ui32, gm, 4>
      %three = arith.constant 3 : i32
      %three_u32 = builtin.unrealized_conversion_cast %three : i32 to ui32
      %old = tla.simd_atomic_add %p, %three_u32 : !tla.ptr<ui32, gm, 4>, ui32 -> ui32
      tla.scalar_store %dst[%c0], %old : !gm_u32, ui32
    }) : () -> ()
    return
  }

  func.func @atomic_add_f32(%dst_memref: memref<8xf32, #hivm.address_space<gm>>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c8 = arith.constant 8 : index
    %dst = tla.tensor_desc %dst_memref shape [%c1, %c8, %c1, %c1] stride [%c8, %c1, %c1, %c1] origin_shape [%c1, %c8] coord [%c0, %c0] : memref<8xf32, #hivm.address_space<gm>> -> !gm_f32
    "tla.vector"() ({
      %p = tla.tensor_ptr %dst : !gm_f32 -> !tla.ptr<f32, gm, 4>
      %half = arith.constant 5.000000e-01 : f32
      %old = tla.simd_atomic_add %p, %half : !tla.ptr<f32, gm, 4>, f32 -> f32
      tla.scalar_store %dst[%c0], %old : !gm_f32, f32
    }) : () -> ()
    return
  }

  func.func @atomic_add_u64_offset(%dst_memref: memref<8xui64, #hivm.address_space<gm>>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c8 = arith.constant 8 : index
    %dst = tla.tensor_desc %dst_memref shape [%c1, %c8, %c1, %c1] stride [%c8, %c1, %c1, %c1] origin_shape [%c1, %c8] coord [%c0, %c0] : memref<8xui64, #hivm.address_space<gm>> -> !gm_u64
    "tla.vector"() ({
      %base = tla.tensor_ptr %dst : !gm_u64 -> !tla.ptr<ui64, gm, 8>
      %p = tla.ptr_add %base, %c2 : !tla.ptr<ui64, gm, 8>, index -> !tla.ptr<ui64, gm, 8>
      %one = arith.constant 1 : i64
      %one_u64 = builtin.unrealized_conversion_cast %one : i64 to ui64
      %old = tla.simd_atomic_add %p, %one_u64 : !tla.ptr<ui64, gm, 8>, ui64 -> ui64
      tla.scalar_store %dst[%c2], %old : !gm_u64, ui64
    }) : () -> ()
    return
  }

  func.func @atomic_exch_u32(%dst_memref: memref<8xui32, #hivm.address_space<gm>>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c8 = arith.constant 8 : index
    %dst = tla.tensor_desc %dst_memref shape [%c1, %c8, %c1, %c1] stride [%c8, %c1, %c1, %c1] origin_shape [%c1, %c8] coord [%c0, %c0] : memref<8xui32, #hivm.address_space<gm>> -> !gm_u32
    "tla.vector"() ({
      %p = tla.tensor_ptr %dst : !gm_u32 -> !tla.ptr<ui32, gm, 4>
      %seven = arith.constant 7 : i32
      %seven_u32 = builtin.unrealized_conversion_cast %seven : i32 to ui32
      %old = tla.simd_atomic_exch %p, %seven_u32 : !tla.ptr<ui32, gm, 4>, ui32 -> ui32
      tla.scalar_store %dst[%c0], %old : !gm_u32, ui32
    }) : () -> ()
    return
  }

  func.func @atomic_exch_u64(%dst_memref: memref<8xui64, #hivm.address_space<gm>>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c8 = arith.constant 8 : index
    %dst = tla.tensor_desc %dst_memref shape [%c1, %c8, %c1, %c1] stride [%c8, %c1, %c1, %c1] origin_shape [%c1, %c8] coord [%c0, %c0] : memref<8xui64, #hivm.address_space<gm>> -> !gm_u64
    "tla.vector"() ({
      %p = tla.tensor_ptr %dst : !gm_u64 -> !tla.ptr<ui64, gm, 8>
      %five = arith.constant 5 : i64
      %five_u64 = builtin.unrealized_conversion_cast %five : i64 to ui64
      %old = tla.simd_atomic_exch %p, %five_u64 : !tla.ptr<ui64, gm, 8>, ui64 -> ui64
      tla.scalar_store %dst[%c0], %old : !gm_u64, ui64
    }) : () -> ()
    return
  }

  "tla.func"() ({
  ^bb0(%dst: !gm_u32):
    "tla.cube"() ({
      %p = tla.tensor_ptr %dst : !gm_u32 -> !tla.ptr<ui32, gm, 4>
      %one = arith.constant 1 : i32
      %one_u32 = builtin.unrealized_conversion_cast %one : i32 to ui32
      %old = tla.simd_atomic_add %p, %one_u32 : !tla.ptr<ui32, gm, 4>, ui32 -> ui32
    }) : () -> ()
    "tla.return"() : () -> ()
  }) {function_type = (!gm_u32) -> (), sym_name = "atomic_add_u32_cube"} : () -> ()
}
