// RUN: %tla_compile %s --mlir-print-ir-after=tla-vector-region-simd -o %t 2>&1 | %filecheck %s
//
// Float16 register store to an int4 tile packs through DIST_PACK4_B32. The
// destination view is int4x2_t (one byte per lane).

!s_f16 = !tla.tensor<!tla.layout<!tla.shape<128>, !tla.stride<1>, !tla.shape<128>, RowMajor>, !tla.coord<0>, !tla.ptr<f16, ub, 32>>
!d_i4 = !tla.tensor<!tla.layout<!tla.shape<256>, !tla.stride<1>, !tla.shape<256>, RowMajor>, !tla.coord<0>, !tla.ptr<!tla.i4x2, ub, 32>>

module {
  func.func @store_f16_to_i4_pack4(%sm: memref<128xf16, #hivm.address_space<ub>>, %dm: memref<256xi8, #hivm.address_space<ub>>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %cs = arith.constant 128 : index
    %cd = arith.constant 256 : index
    %s = tla.tensor_desc %sm shape [%c1, %cs, %c1, %c1] stride [%cs, %c1, %c1, %c1] origin_shape [%c1, %cs] coord [%c0, %c0] : memref<128xf16, #hivm.address_space<ub>> -> !s_f16
    %d = tla.tensor_desc %dm shape [%c1, %cd, %c1, %c1] stride [%cd, %c1, %c1, %c1] origin_shape [%c1, %cd] coord [%c0, %c0] : memref<256xi8, #hivm.address_space<ub>> -> !d_i4
    "tla.vec.func"() ({
      %v = tla.load %s : !s_f16 -> !tla.vector<128xf16>
      tla.store %d, %v {store_dist = #tla.store_dist<pack4_b32>} : !d_i4, !tla.vector<128xf16>
    }) {mode = "simd"} : () -> ()
    return
  }
}

// CHECK-LABEL: func.func private @vector_region_0
// CHECK: ave.hir.vload <NORM>
// CHECK: ave.hir.masked_store <PK4_B32>
