// RUN: %tla_compile %s -o - | %filecheck %s

!src = !tla.tensor<!tla.layout<!tla.shape<256>, !tla.stride<1>, !tla.shape<256>, RowMajor>, !tla.coord<0>, !tla.ptr<!tla.i4x2, ub, 32>>
!dst = !tla.tensor<!tla.layout<!tla.shape<256>, !tla.stride<1>, !tla.shape<256>, RowMajor>, !tla.coord<0>, !tla.ptr<!tla.i4x2, ub, 32>>

module {
  func.func @int4x2_flat_offset_load_store(%sm: memref<256xi8, #hivm.address_space<ub>>, %dm: memref<256xi8, #hivm.address_space<ub>>) {
    %c0 = arith.constant 0 : index
    %c128 = arith.constant 128 : index
    %c1 = arith.constant 1 : index
    %cs = arith.constant 256 : index
    %s = tla.tensor_desc %sm shape [%c1, %cs, %c1, %c1] stride [%cs, %c1, %c1, %c1] origin_shape [%c1, %cs] coord [%c0, %c128] : memref<256xi8, #hivm.address_space<ub>> -> !src
    %d = tla.tensor_desc %dm shape [%c1, %cs, %c1, %c1] stride [%cs, %c1, %c1, %c1] origin_shape [%c1, %cs] coord [%c0, %c128] : memref<256xi8, #hivm.address_space<ub>> -> !dst
    "tla.vec.func"() ({
      %v = tla.load %s : !src -> !tla.vector<256x!tla.i4x2>
      tla.store %d, %v : !dst, !tla.vector<256x!tla.i4x2>
    }) {mode = "simd"} : () -> ()
    return
  }
}

// CHECK-LABEL: func.func private @vector_region_0
// CHECK: llvm.mlir.constant(128 : index)
// CHECK: memref.reinterpret_cast %arg0 to offset: [%{{.*}}]
// CHECK: memref.reinterpret_cast %arg1 to offset: [%{{.*}}]
// CHECK: hivm_regbaseintrins.intr.hivm.vldsx1.v256s8
// CHECK: hivm_regbaseintrins.intr.hivm.vstsx1.v256s8
