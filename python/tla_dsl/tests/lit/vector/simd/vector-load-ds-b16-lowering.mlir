// RUN: %tla_compile %s --mlir-print-ir-after=tla-vector-region-simd -o %t 2>&1 | %filecheck %s --check-prefix=AVE
// RUN: %filecheck %s --check-prefix=INTR < %t

!s_i16 = !tla.tensor<!tla.layout<!tla.shape<256>, !tla.stride<1>, !tla.shape<256>, RowMajor>, !tla.coord<0>, !tla.ptr<i16, ub, 2>>
!d_i16 = !tla.tensor<!tla.layout<!tla.shape<128>, !tla.stride<1>, !tla.shape<128>, RowMajor>, !tla.coord<0>, !tla.ptr<i16, ub, 2>>
!s_ui16 = !tla.tensor<!tla.layout<!tla.shape<256>, !tla.stride<1>, !tla.shape<256>, RowMajor>, !tla.coord<0>, !tla.ptr<ui16, ub, 2>>
!d_ui16 = !tla.tensor<!tla.layout<!tla.shape<128>, !tla.stride<1>, !tla.shape<128>, RowMajor>, !tla.coord<0>, !tla.ptr<ui16, ub, 2>>
!s_f16 = !tla.tensor<!tla.layout<!tla.shape<256>, !tla.stride<1>, !tla.shape<256>, RowMajor>, !tla.coord<0>, !tla.ptr<f16, ub, 2>>
!d_f16 = !tla.tensor<!tla.layout<!tla.shape<128>, !tla.stride<1>, !tla.shape<128>, RowMajor>, !tla.coord<0>, !tla.ptr<f16, ub, 2>>
!s_bf16 = !tla.tensor<!tla.layout<!tla.shape<256>, !tla.stride<1>, !tla.shape<256>, RowMajor>, !tla.coord<0>, !tla.ptr<bf16, ub, 2>>
!d_bf16 = !tla.tensor<!tla.layout<!tla.shape<128>, !tla.stride<1>, !tla.shape<128>, RowMajor>, !tla.coord<0>, !tla.ptr<bf16, ub, 2>>

module {
  func.func @load_ds_b16_i16(%s: memref<256xi16, #hivm.address_space<ub>>, %d: memref<128xi16, #hivm.address_space<ub>>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %cn = arith.constant 256 : index
    %cv = arith.constant 128 : index
    %src = tla.tensor_desc %s shape [%c1, %cn, %c1, %c1] stride [%cn, %c1, %c1, %c1] origin_shape [%c1, %cn] coord [%c0, %c0] : memref<256xi16, #hivm.address_space<ub>> -> !s_i16
    %dst = tla.tensor_desc %d shape [%c1, %cv, %c1, %c1] stride [%cv, %c1, %c1, %c1] origin_shape [%c1, %cv] coord [%c0, %c0] : memref<128xi16, #hivm.address_space<ub>> -> !d_i16
    "tla.vec.func"() ({
      %v = tla.load %src {load_dist = #tla.load_dist<ds_b16>} : !s_i16 -> !tla.vector<128xi16>
      tla.store %dst, %v : !d_i16, !tla.vector<128xi16>
    }) : () -> ()
    return
  }

  func.func @load_ds_b16_ui16(%s: memref<256xui16, #hivm.address_space<ub>>, %d: memref<128xui16, #hivm.address_space<ub>>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %cn = arith.constant 256 : index
    %cv = arith.constant 128 : index
    %src = tla.tensor_desc %s shape [%c1, %cn, %c1, %c1] stride [%cn, %c1, %c1, %c1] origin_shape [%c1, %cn] coord [%c0, %c0] : memref<256xui16, #hivm.address_space<ub>> -> !s_ui16
    %dst = tla.tensor_desc %d shape [%c1, %cv, %c1, %c1] stride [%cv, %c1, %c1, %c1] origin_shape [%c1, %cv] coord [%c0, %c0] : memref<128xui16, #hivm.address_space<ub>> -> !d_ui16
    "tla.vec.func"() ({
      %v = tla.load %src {load_dist = #tla.load_dist<ds_b16>} : !s_ui16 -> !tla.vector<128xui16>
      tla.store %dst, %v : !d_ui16, !tla.vector<128xui16>
    }) : () -> ()
    return
  }

  func.func @load_ds_b16_f16(%s: memref<256xf16, #hivm.address_space<ub>>, %d: memref<128xf16, #hivm.address_space<ub>>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %cn = arith.constant 256 : index
    %cv = arith.constant 128 : index
    %src = tla.tensor_desc %s shape [%c1, %cn, %c1, %c1] stride [%cn, %c1, %c1, %c1] origin_shape [%c1, %cn] coord [%c0, %c0] : memref<256xf16, #hivm.address_space<ub>> -> !s_f16
    %dst = tla.tensor_desc %d shape [%c1, %cv, %c1, %c1] stride [%cv, %c1, %c1, %c1] origin_shape [%c1, %cv] coord [%c0, %c0] : memref<128xf16, #hivm.address_space<ub>> -> !d_f16
    "tla.vec.func"() ({
      %v = tla.load %src {load_dist = #tla.load_dist<ds_b16>} : !s_f16 -> !tla.vector<128xf16>
      tla.store %dst, %v : !d_f16, !tla.vector<128xf16>
    }) : () -> ()
    return
  }

  func.func @load_ds_b16_bf16(%s: memref<256xbf16, #hivm.address_space<ub>>, %d: memref<128xbf16, #hivm.address_space<ub>>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %cn = arith.constant 256 : index
    %cv = arith.constant 128 : index
    %src = tla.tensor_desc %s shape [%c1, %cn, %c1, %c1] stride [%cn, %c1, %c1, %c1] origin_shape [%c1, %cn] coord [%c0, %c0] : memref<256xbf16, #hivm.address_space<ub>> -> !s_bf16
    %dst = tla.tensor_desc %d shape [%c1, %cv, %c1, %c1] stride [%cv, %c1, %c1, %c1] origin_shape [%c1, %cv] coord [%c0, %c0] : memref<128xbf16, #hivm.address_space<ub>> -> !d_bf16
    "tla.vec.func"() ({
      %v = tla.load %src {load_dist = #tla.load_dist<ds_b16>} : !s_bf16 -> !tla.vector<128xbf16>
      tla.store %dst, %v : !d_bf16, !tla.vector<128xbf16>
    }) : () -> ()
    return
  }

}

// AVE-LABEL: func.func private @vector_region_
// AVE: ave.hir.vload <DS_B16>
// AVE-SAME: into vector<128xi16>
// AVE-LABEL: func.func private @vector_region_
// AVE: ave.hir.vload <DS_B16>
// AVE-SAME: into vector<128xi16>
// AVE-LABEL: func.func private @vector_region_
// AVE: ave.hir.vload <DS_B16>
// AVE-SAME: into vector<128xf16>
// AVE-LABEL: func.func private @vector_region_
// AVE: ave.hir.vload <DS_B16>
// AVE-SAME: into vector<128xbf16>

// INTR: hivm_regbaseintrins.intr.hivm.vldsx1.v128{{[su]}}16
// INTR: hivm_regbaseintrins.intr.hivm.vldsx1.v128{{[su]}}16
// INTR: hivm_regbaseintrins.intr.hivm.vldsx1.v128f16
// INTR: hivm_regbaseintrins.intr.hivm.vldsx1.v128bf16
