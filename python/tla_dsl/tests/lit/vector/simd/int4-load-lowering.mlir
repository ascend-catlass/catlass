// RUN: %tla_compile %s --mlir-print-ir-after=tla-vector-region-simd -o %t 2>&1 | %filecheck %s
//
// Int4x2 logical vectors lower through int4x2_t byte registers. NORM is a plain
// packed load; UNPACK4_B8 is the byte-expanding distribution that feeds a
// register cast.

!norm_src = !tla.tensor<!tla.layout<!tla.shape<256>, !tla.stride<1>, !tla.shape<256>, RowMajor>, !tla.coord<0>, !tla.ptr<!tla.i4x2, ub, 32>>
!norm_dst = !tla.tensor<!tla.layout<!tla.shape<256>, !tla.stride<1>, !tla.shape<256>, RowMajor>, !tla.coord<0>, !tla.ptr<!tla.i4x2, ub, 32>>
!unpk_src = !tla.tensor<!tla.layout<!tla.shape<64>, !tla.stride<1>, !tla.shape<64>, RowMajor>, !tla.coord<0>, !tla.ptr<!tla.i4x2, ub, 32>>
!unpk_dst = !tla.tensor<!tla.layout<!tla.shape<256>, !tla.stride<1>, !tla.shape<256>, RowMajor>, !tla.coord<0>, !tla.ptr<!tla.i4x2, ub, 32>>

module {
  func.func @int4_norm_load_store(%sm: memref<256xi8, #hivm.address_space<ub>>, %dm: memref<256xi8, #hivm.address_space<ub>>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %cs = arith.constant 256 : index
    %s = tla.tensor_desc %sm shape [%c1, %cs, %c1, %c1] stride [%cs, %c1, %c1, %c1] origin_shape [%c1, %cs] coord [%c0, %c0] : memref<256xi8, #hivm.address_space<ub>> -> !norm_src
    %d = tla.tensor_desc %dm shape [%c1, %cs, %c1, %c1] stride [%cs, %c1, %c1, %c1] origin_shape [%c1, %cs] coord [%c0, %c0] : memref<256xi8, #hivm.address_space<ub>> -> !norm_dst
    "tla.vec.func"() ({
      %v = tla.load %s : !norm_src -> !tla.vector<256x!tla.i4x2>
      tla.store %d, %v : !norm_dst, !tla.vector<256x!tla.i4x2>
    }) {mode = "simd"} : () -> ()
    return
  }

  func.func @int4_unpack4_load(%sm: memref<64xi8, #hivm.address_space<ub>>, %dm: memref<256xi8, #hivm.address_space<ub>>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %cs = arith.constant 64 : index
    %cd = arith.constant 256 : index
    %s = tla.tensor_desc %sm shape [%c1, %cs, %c1, %c1] stride [%cs, %c1, %c1, %c1] origin_shape [%c1, %cs] coord [%c0, %c0] : memref<64xi8, #hivm.address_space<ub>> -> !unpk_src
    %d = tla.tensor_desc %dm shape [%c1, %cd, %c1, %c1] stride [%cd, %c1, %c1, %c1] origin_shape [%c1, %cd] coord [%c0, %c0] : memref<256xi8, #hivm.address_space<ub>> -> !unpk_dst
    "tla.vec.func"() ({
      %v = tla.load %s {load_dist = #tla.load_dist<unpack4_b8>} : !unpk_src -> !tla.vector<256x!tla.i4x2>
      tla.store %d, %v : !unpk_dst, !tla.vector<256x!tla.i4x2>
    }) {mode = "simd"} : () -> ()
    return
  }
}

// CHECK-LABEL: func.func private @vector_region_0
// CHECK: ave.hir.vload <NORM>
// CHECK: ave.hir.masked_store <NORM_B8>
// CHECK-LABEL: func.func private @vector_region_1
// CHECK: ave.hir.vload <UNPK4_B8>
// CHECK: ave.hir.masked_store <NORM_B8>
