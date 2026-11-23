// RUN: %tla_compile %s --mlir-print-ir-after=tla-vector-region-simd -o %t 2>&1 | %filecheck %s
// E8M0 is move-only b8 transport; INTLV_B8 is allowed alongside DIST_NORM.

!src = !tla.tensor<!tla.layout<!tla.shape<256>, !tla.stride<1>, !tla.shape<256>, RowMajor>, !tla.coord<0>, !tla.ptr<!tla.f8e8m0, ub, 1>>
!dst = !tla.tensor<!tla.layout<!tla.shape<512>, !tla.stride<1>, !tla.shape<512>, RowMajor>, !tla.coord<0>, !tla.ptr<!tla.f8e8m0, ub, 1>>

module {
  func.func @store_intlv_b8_e8m0(
      %even: memref<256xi8, #hivm.address_space<ub>>,
      %odd: memref<256xi8, #hivm.address_space<ub>>,
      %dest: memref<512xi8, #hivm.address_space<ub>>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c256 = arith.constant 256 : index
    %c512 = arith.constant 512 : index
    %e = tla.tensor_desc %even shape [%c1, %c256, %c1, %c1] stride [%c256, %c1, %c1, %c1] origin_shape [%c1, %c256] coord [%c0, %c0] : memref<256xi8, #hivm.address_space<ub>> -> !src
    %o = tla.tensor_desc %odd shape [%c1, %c256, %c1, %c1] stride [%c256, %c1, %c1, %c1] origin_shape [%c1, %c256] coord [%c0, %c0] : memref<256xi8, #hivm.address_space<ub>> -> !src
    %d = tla.tensor_desc %dest shape [%c1, %c512, %c1, %c1] stride [%c512, %c1, %c1, %c1] origin_shape [%c1, %c512] coord [%c0, %c0] : memref<512xi8, #hivm.address_space<ub>> -> !dst
    "tla.vec.func"() ({
      %shape = "tla.make_shape"() : () -> !tla.shape<256>
      %dshape = "tla.make_shape"() : () -> !tla.shape<512>
      %coord = "tla.make_coord"() : () -> !tla.coord<0>
      %et = "tla.tile_view"(%e, %shape, %coord) : (!src, !tla.shape<256>, !tla.coord<0>) -> !src
      %ot = "tla.tile_view"(%o, %shape, %coord) : (!src, !tla.shape<256>, !tla.coord<0>) -> !src
      %dt = "tla.tile_view"(%d, %dshape, %coord) : (!dst, !tla.shape<512>, !tla.coord<0>) -> !dst
      %v0 = tla.load %et : !src -> !tla.vector<256x!tla.f8e8m0>
      %v1 = tla.load %ot : !src -> !tla.vector<256x!tla.f8e8m0>
      tla.store %dt, %v0, %v1 {store_dist = #tla.store_dist<intlv_b8>} : !dst, !tla.vector<256x!tla.f8e8m0>, !tla.vector<256x!tla.f8e8m0>
    }) : () -> ()
    return
  }
}

// CHECK-LABEL: func.func private @vector_region_
// CHECK: "hivm_regbaseintrins.intr.hivm.vstsx2.v256b8"
// CHECK-NOT: tla.store
