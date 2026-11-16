// RUN: %tla_compile %s --mlir-print-ir-after=tla-vector-region-simd -o %t 2>&1 | %filecheck %s --check-prefix=AVE
// RUN: %filecheck %s --check-prefix=INTR < %t

!s_i8 = !tla.tensor<!tla.layout<!tla.shape<128>, !tla.stride<1>, !tla.shape<128>, RowMajor>, !tla.coord<0>, !tla.ptr<i8, ub, 1>>
!d_i8 = !tla.tensor<!tla.layout<!tla.shape<256>, !tla.stride<1>, !tla.shape<256>, RowMajor>, !tla.coord<0>, !tla.ptr<i8, ub, 1>>
!s_ui8 = !tla.tensor<!tla.layout<!tla.shape<128>, !tla.stride<1>, !tla.shape<128>, RowMajor>, !tla.coord<0>, !tla.ptr<ui8, ub, 1>>
!d_ui8 = !tla.tensor<!tla.layout<!tla.shape<256>, !tla.stride<1>, !tla.shape<256>, RowMajor>, !tla.coord<0>, !tla.ptr<ui8, ub, 1>>
!s_f8e4m3fn = !tla.tensor<!tla.layout<!tla.shape<128>, !tla.stride<1>, !tla.shape<128>, RowMajor>, !tla.coord<0>, !tla.ptr<f8E4M3FN, ub, 1>>
!d_f8e4m3fn = !tla.tensor<!tla.layout<!tla.shape<256>, !tla.stride<1>, !tla.shape<256>, RowMajor>, !tla.coord<0>, !tla.ptr<f8E4M3FN, ub, 1>>
!s_f8e5m2 = !tla.tensor<!tla.layout<!tla.shape<128>, !tla.stride<1>, !tla.shape<128>, RowMajor>, !tla.coord<0>, !tla.ptr<f8E5M2, ub, 1>>
!d_f8e5m2 = !tla.tensor<!tla.layout<!tla.shape<256>, !tla.stride<1>, !tla.shape<256>, RowMajor>, !tla.coord<0>, !tla.ptr<f8E5M2, ub, 1>>

module {
  func.func @load_unpack_b8_i8(%s: memref<128xi8, #hivm.address_space<ub>>, %d: memref<256xi8, #hivm.address_space<ub>>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %cn = arith.constant 128 : index
    %cv = arith.constant 256 : index
    %src = tla.tensor_desc %s shape [%c1, %cn, %c1, %c1] stride [%cn, %c1, %c1, %c1] origin_shape [%c1, %cn] coord [%c0, %c0] : memref<128xi8, #hivm.address_space<ub>> -> !s_i8
    %dst = tla.tensor_desc %d shape [%c1, %cv, %c1, %c1] stride [%cv, %c1, %c1, %c1] origin_shape [%c1, %cv] coord [%c0, %c0] : memref<256xi8, #hivm.address_space<ub>> -> !d_i8
    "tla.vec.func"() ({
      %v = tla.load %src {load_dist = #tla.load_dist<unpack_b8>} : !s_i8 -> !tla.vector<256xi8>
      tla.store %dst, %v : !d_i8, !tla.vector<256xi8>
    }) : () -> ()
    return
  }

  func.func @load_unpack_b8_ui8(%s: memref<128xui8, #hivm.address_space<ub>>, %d: memref<256xui8, #hivm.address_space<ub>>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %cn = arith.constant 128 : index
    %cv = arith.constant 256 : index
    %src = tla.tensor_desc %s shape [%c1, %cn, %c1, %c1] stride [%cn, %c1, %c1, %c1] origin_shape [%c1, %cn] coord [%c0, %c0] : memref<128xui8, #hivm.address_space<ub>> -> !s_ui8
    %dst = tla.tensor_desc %d shape [%c1, %cv, %c1, %c1] stride [%cv, %c1, %c1, %c1] origin_shape [%c1, %cv] coord [%c0, %c0] : memref<256xui8, #hivm.address_space<ub>> -> !d_ui8
    "tla.vec.func"() ({
      %v = tla.load %src {load_dist = #tla.load_dist<unpack_b8>} : !s_ui8 -> !tla.vector<256xui8>
      tla.store %dst, %v : !d_ui8, !tla.vector<256xui8>
    }) : () -> ()
    return
  }

  func.func @load_unpack_b8_f8e4m3fn(%s: memref<128xf8E4M3FN, #hivm.address_space<ub>>, %d: memref<256xf8E4M3FN, #hivm.address_space<ub>>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %cn = arith.constant 128 : index
    %cv = arith.constant 256 : index
    %src = tla.tensor_desc %s shape [%c1, %cn, %c1, %c1] stride [%cn, %c1, %c1, %c1] origin_shape [%c1, %cn] coord [%c0, %c0] : memref<128xf8E4M3FN, #hivm.address_space<ub>> -> !s_f8e4m3fn
    %dst = tla.tensor_desc %d shape [%c1, %cv, %c1, %c1] stride [%cv, %c1, %c1, %c1] origin_shape [%c1, %cv] coord [%c0, %c0] : memref<256xf8E4M3FN, #hivm.address_space<ub>> -> !d_f8e4m3fn
    "tla.vec.func"() ({
      %v = tla.load %src {load_dist = #tla.load_dist<unpack_b8>} : !s_f8e4m3fn -> !tla.vector<256xf8E4M3FN>
      tla.store %dst, %v : !d_f8e4m3fn, !tla.vector<256xf8E4M3FN>
    }) : () -> ()
    return
  }

  func.func @load_unpack_b8_f8e5m2(%s: memref<128xf8E5M2, #hivm.address_space<ub>>, %d: memref<256xf8E5M2, #hivm.address_space<ub>>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %cn = arith.constant 128 : index
    %cv = arith.constant 256 : index
    %src = tla.tensor_desc %s shape [%c1, %cn, %c1, %c1] stride [%cn, %c1, %c1, %c1] origin_shape [%c1, %cn] coord [%c0, %c0] : memref<128xf8E5M2, #hivm.address_space<ub>> -> !s_f8e5m2
    %dst = tla.tensor_desc %d shape [%c1, %cv, %c1, %c1] stride [%cv, %c1, %c1, %c1] origin_shape [%c1, %cv] coord [%c0, %c0] : memref<256xf8E5M2, #hivm.address_space<ub>> -> !d_f8e5m2
    "tla.vec.func"() ({
      %v = tla.load %src {load_dist = #tla.load_dist<unpack_b8>} : !s_f8e5m2 -> !tla.vector<256xf8E5M2>
      tla.store %dst, %v : !d_f8e5m2, !tla.vector<256xf8E5M2>
    }) : () -> ()
    return
  }

}

// AVE-LABEL: func.func private @vector_region_
// AVE: ave.hir.vload <UNPK_B8>
// AVE-SAME: into vector<256xi8>
// AVE-LABEL: func.func private @vector_region_
// AVE: ave.hir.vload <UNPK_B8>
// AVE-SAME: into vector<256xi8>
// AVE-LABEL: func.func private @vector_region_
// AVE: ave.hir.vload <UNPK_B8>
// AVE-SAME: into vector<256xf8E4M3FN>
// AVE-LABEL: func.func private @vector_region_
// AVE: ave.hir.vload <UNPK_B8>
// AVE-SAME: into vector<256xf8E5M2>

// INTR: hivm_regbaseintrins.intr.hivm.vldsx1.v256{{[su]}}8
// INTR: hivm_regbaseintrins.intr.hivm.vldsx1.v256{{[su]}}8
// INTR: hivm_regbaseintrins.intr.hivm.vldsx1.v256f8e4m3
// INTR: hivm_regbaseintrins.intr.hivm.vldsx1.v256f8e5m2
