// RUN: %tla_compile %s --mlir-print-ir-after=tla-vector-region-simd -o %t 2>&1 | %filecheck %s

!source = !tla.tensor<!tla.layout<!tla.shape<128>, !tla.stride<1>, !tla.shape<128>, RowMajor>, !tla.coord<0>, !tla.ptr<bf16, ub, 32>>
!dest = !tla.tensor<!tla.layout<!tla.shape<256>, !tla.stride<1>, !tla.shape<256>, RowMajor>, !tla.coord<0>, !tla.ptr<!tla.f8e8m0, ub, 32>>

func.func @extract_exponents(%in: memref<128xbf16, #hivm.address_space<ub>>,
                            %out_odd: memref<256xi8, #hivm.address_space<ub>>,
                            %out_even: memref<256xi8, #hivm.address_space<ub>>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c128 = arith.constant 128 : index
  %c256 = arith.constant 256 : index
  %s = tla.tensor_desc %in shape [%c1, %c128, %c1, %c1] stride [%c128, %c1, %c1, %c1] origin_shape [%c1, %c128] coord [%c0, %c0] : memref<128xbf16, #hivm.address_space<ub>> -> !source
  %d_odd = tla.tensor_desc %out_odd shape [%c1, %c256, %c1, %c1] stride [%c256, %c1, %c1, %c1] origin_shape [%c1, %c256] coord [%c0, %c0] : memref<256xi8, #hivm.address_space<ub>> -> !dest
  %d_even = tla.tensor_desc %out_even shape [%c1, %c256, %c1, %c1] stride [%c256, %c1, %c1, %c1] origin_shape [%c1, %c256] coord [%c0, %c0] : memref<256xi8, #hivm.address_space<ub>> -> !dest
  "tla.vec.func"() ({
    %v = tla.load %s : !source -> !tla.vector<128xbf16>
    %p = "tla.create_mask"() {pattern = "VL3", dtype = bf16} : () -> !tla.mask<128>
    %odd = tla.cast %v, [1, 0, 4] : !tla.vector<128xbf16> -> !tla.vector<?x!tla.f8e8m0>
    %even = tla.cast %v, [0, 0, 4] mask %p : !tla.vector<128xbf16> mask !tla.mask<128> -> !tla.vector<?x!tla.f8e8m0>
    tla.store %d_odd, %odd : !dest, !tla.vector<?x!tla.f8e8m0>
    tla.store %d_even, %even : !dest, !tla.vector<?x!tla.f8e8m0>
  }) {mode = "simd"} : () -> ()
  return
}

// CHECK: ave.hir.vshls
// CHECK: ave.hir.vshrs
// CHECK: ave.hir.vtrunci {{.*}}part_odd{{.*}}ui2ui
// CHECK: %[[SHIFT:.*]] = ave.hir.vshls {{.*}}, %[[MASK:[^, ]+]], {{.*}} : vector<128xi16>
// CHECK: %[[EXP:.*]] = ave.hir.vshrs %[[SHIFT]], {{.*}}, %[[MASK]],
// CHECK: %[[INF_BITS:[^ ]+]] = arith.constant -256 : i16
// CHECK: %[[IS_INF:.*]] = ave.hir.vcmps <EQ> %[[SHIFT]], %[[INF_BITS]], %[[MASK]]
// CHECK: %[[MAX_FINITE:[^ ]+]] = arith.constant 254 : i16
// CHECK: %[[FINITE_LANES:.*]] = ave.hir.scalar_broadcast %[[MAX_FINITE]]
// CHECK: %[[ENCODED:.*]] = ave.hir.vsel %[[IS_INF]], %[[FINITE_LANES]], %[[EXP]]
// CHECK: ave.hir.vtrunci %[[ENCODED]], true, %[[MASK]] {{.*}}part_even{{.*}}ui2ui
