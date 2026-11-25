// RUN: %tla_compile %s --emit=mlir --mlir-print-ir-after=tla-vector-region-simd -o /dev/null 2>&1 | %filecheck %s --check-prefix=OUTLINE
// RUN: %tla_compile %s --emit=mlir -o - | %filecheck %s --check-prefix=FINAL

// Atomics belong to the scalar part of a SIMD region. The atom stays a tla op
// while the region's vector work is outlined into a helper, and the arch-op
// pass lowers it afterwards in the parent function -- never inside the helper.

// OUTLINE-LABEL: func.func @atomic_with_vector_store
// OUTLINE: tla.simd_atomic_add
// OUTLINE-LABEL: func.func private @vector_region_
// OUTLINE-NOT: tla.simd_atomic_add
// OUTLINE-NOT: llvm.call_intrinsic

// FINAL-LABEL: func.func @atomic_with_vector_store
// FINAL: llvm.hivm.atom.ADD.G.s32
// FINAL-NOT: tla.simd_atomic

!gm_i32 = !tla.tensor<!tla.layout<!tla.shape<8>, !tla.stride<1>, !tla.shape<8>, RowMajor>, !tla.coord<0>, !tla.ptr<i32, gm, 4>>
!ub_f32 = !tla.tensor<!tla.layout<!tla.shape<64>, !tla.stride<1>, !tla.shape<64>, RowMajor>, !tla.coord<0>, !tla.ptr<f32, ub, 4>>

module {
  func.func @atomic_with_vector_store(
      %dst_memref: memref<8xi32, #hivm.address_space<gm>>,
      %ub_memref: memref<64xf32, #hivm.address_space<ub>>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c8 = arith.constant 8 : index
    %c64 = arith.constant 64 : index
    %dst = tla.tensor_desc %dst_memref shape [%c1, %c8, %c1, %c1] stride [%c8, %c1, %c1, %c1] origin_shape [%c1, %c8] coord [%c0, %c0] : memref<8xi32, #hivm.address_space<gm>> -> !gm_i32
    %ub = tla.tensor_desc %ub_memref shape [%c1, %c64, %c1, %c1] stride [%c64, %c1, %c1, %c1] origin_shape [%c1, %c64] coord [%c0, %c0] : memref<64xf32, #hivm.address_space<ub>> -> !ub_f32
    "tla.vector"() ({
      %p = tla.tensor_ptr %dst : !gm_i32 -> !tla.ptr<i32, gm, 4>
      %one = arith.constant 1 : i32
      %old = tla.simd_atomic_add %p, %one : !tla.ptr<i32, gm, 4>, i32 -> i32
      tla.scalar_store %dst[%c0], %old : !gm_i32, i32
      "tla.vec.func"() ({
        %shape = "tla.make_shape"() : () -> !tla.shape<64>
        %coord = "tla.make_coord"() : () -> !tla.coord<0>
        %tile = "tla.tile_view"(%ub, %shape, %coord) : (!ub_f32, !tla.shape<64>, !tla.coord<0>) -> !ub_f32
        %f = arith.constant 1.000000e+00 : f32
        %v = "tla.full"(%f) : (f32) -> !tla.vector<64xf32>
        tla.store %tile, %v : !ub_f32, !tla.vector<64xf32>
      }) {mode = "simd"} : () -> ()
    }) : () -> ()
    return
  }
}
