// RUN: %tla_compile %s --mlir-print-ir-after=tla-vector-region-simd -o %t 2>&1 | %filecheck %s --check-prefix=AVE
// RUN: %filecheck %s --check-prefix=INTR < %t

!s_i32 = !tla.tensor<!tla.layout<!tla.shape<32>, !tla.stride<1>, !tla.shape<32>, RowMajor>, !tla.coord<0>, !tla.ptr<i32, ub, 4>>
!d_i32 = !tla.tensor<!tla.layout<!tla.shape<64>, !tla.stride<1>, !tla.shape<64>, RowMajor>, !tla.coord<0>, !tla.ptr<i32, ub, 4>>
!s_ui32 = !tla.tensor<!tla.layout<!tla.shape<32>, !tla.stride<1>, !tla.shape<32>, RowMajor>, !tla.coord<0>, !tla.ptr<ui32, ub, 4>>
!d_ui32 = !tla.tensor<!tla.layout<!tla.shape<64>, !tla.stride<1>, !tla.shape<64>, RowMajor>, !tla.coord<0>, !tla.ptr<ui32, ub, 4>>
!s_f32 = !tla.tensor<!tla.layout<!tla.shape<32>, !tla.stride<1>, !tla.shape<32>, RowMajor>, !tla.coord<0>, !tla.ptr<f32, ub, 4>>
!d_f32 = !tla.tensor<!tla.layout<!tla.shape<64>, !tla.stride<1>, !tla.shape<64>, RowMajor>, !tla.coord<0>, !tla.ptr<f32, ub, 4>>

module {
  func.func @load_unpack_b32_i32(%s: memref<32xi32, #hivm.address_space<ub>>, %d: memref<64xi32, #hivm.address_space<ub>>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %cn = arith.constant 32 : index
    %cv = arith.constant 64 : index
    %src = tla.tensor_desc %s shape [%c1, %cn, %c1, %c1] stride [%cn, %c1, %c1, %c1] origin_shape [%c1, %cn] coord [%c0, %c0] : memref<32xi32, #hivm.address_space<ub>> -> !s_i32
    %dst = tla.tensor_desc %d shape [%c1, %cv, %c1, %c1] stride [%cv, %c1, %c1, %c1] origin_shape [%c1, %cv] coord [%c0, %c0] : memref<64xi32, #hivm.address_space<ub>> -> !d_i32
    "tla.vec.func"() ({
      %v = tla.load %src {load_dist = #tla.load_dist<unpack_b32>} : !s_i32 -> !tla.vector<64xi32>
      tla.store %dst, %v : !d_i32, !tla.vector<64xi32>
    }) : () -> ()
    return
  }

  func.func @load_unpack_b32_ui32(%s: memref<32xui32, #hivm.address_space<ub>>, %d: memref<64xui32, #hivm.address_space<ub>>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %cn = arith.constant 32 : index
    %cv = arith.constant 64 : index
    %src = tla.tensor_desc %s shape [%c1, %cn, %c1, %c1] stride [%cn, %c1, %c1, %c1] origin_shape [%c1, %cn] coord [%c0, %c0] : memref<32xui32, #hivm.address_space<ub>> -> !s_ui32
    %dst = tla.tensor_desc %d shape [%c1, %cv, %c1, %c1] stride [%cv, %c1, %c1, %c1] origin_shape [%c1, %cv] coord [%c0, %c0] : memref<64xui32, #hivm.address_space<ub>> -> !d_ui32
    "tla.vec.func"() ({
      %v = tla.load %src {load_dist = #tla.load_dist<unpack_b32>} : !s_ui32 -> !tla.vector<64xui32>
      tla.store %dst, %v : !d_ui32, !tla.vector<64xui32>
    }) : () -> ()
    return
  }

  func.func @load_unpack_b32_f32(%s: memref<32xf32, #hivm.address_space<ub>>, %d: memref<64xf32, #hivm.address_space<ub>>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %cn = arith.constant 32 : index
    %cv = arith.constant 64 : index
    %src = tla.tensor_desc %s shape [%c1, %cn, %c1, %c1] stride [%cn, %c1, %c1, %c1] origin_shape [%c1, %cn] coord [%c0, %c0] : memref<32xf32, #hivm.address_space<ub>> -> !s_f32
    %dst = tla.tensor_desc %d shape [%c1, %cv, %c1, %c1] stride [%cv, %c1, %c1, %c1] origin_shape [%c1, %cv] coord [%c0, %c0] : memref<64xf32, #hivm.address_space<ub>> -> !d_f32
    "tla.vec.func"() ({
      %v = tla.load %src {load_dist = #tla.load_dist<unpack_b32>} : !s_f32 -> !tla.vector<64xf32>
      tla.store %dst, %v : !d_f32, !tla.vector<64xf32>
    }) : () -> ()
    return
  }

}

// AVE-LABEL: func.func private @vector_region_
// AVE: ave.hir.vload <UNPK_B32>
// AVE-SAME: into vector<64xi32>
// AVE-LABEL: func.func private @vector_region_
// AVE: ave.hir.vload <UNPK_B32>
// AVE-SAME: into vector<64xi32>
// AVE-LABEL: func.func private @vector_region_
// AVE: ave.hir.vload <UNPK_B32>
// AVE-SAME: into vector<64xf32>

// INTR: hivm_regbaseintrins.intr.hivm.vldsx1.v64{{[su]}}32
// INTR: hivm_regbaseintrins.intr.hivm.vldsx1.v64{{[su]}}32
// INTR: hivm_regbaseintrins.intr.hivm.vldsx1.v64f32
