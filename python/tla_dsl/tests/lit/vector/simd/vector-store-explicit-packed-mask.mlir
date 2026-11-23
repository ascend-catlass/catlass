// RUN: %tla_compile %s --mlir-print-ir-after=tla-vector-region-simd -o %t 2>&1 | %filecheck %s

// Explicit predicates preserve the packed-store contract. AscendC PACK_B16 is
// same-type (f16→f16). Omitted-mask geometry must not reject or replace the mask.

!ub_f16 = !tla.tensor<!tla.layout<!tla.shape<128>, !tla.stride<1>, !tla.shape<128>, RowMajor>, !tla.coord<0>, !tla.ptr<f16, ub, 2>>

module {
  func.func @explicit_pack_b16_mask(
      %src_memref: memref<128xf16, #hivm.address_space<ub>>,
      %dst_memref: memref<128xf16, #hivm.address_space<ub>>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c128 = arith.constant 128 : index
    %src = tla.tensor_desc %src_memref shape [%c1, %c128, %c1, %c1] stride [%c128, %c1, %c1, %c1] origin_shape [%c1, %c128] coord [%c0, %c0] : memref<128xf16, #hivm.address_space<ub>> -> !ub_f16
    %dst = tla.tensor_desc %dst_memref shape [%c1, %c128, %c1, %c1] stride [%c128, %c1, %c1, %c1] origin_shape [%c1, %c128] coord [%c0, %c0] : memref<128xf16, #hivm.address_space<ub>> -> !ub_f16
    "tla.vec.func"() ({
      %mask = "tla.create_mask"() {pattern = "H", dtype = f16} : () -> !tla.mask<128>
      %source = tla.load %src : !ub_f16 -> !tla.vector<128xf16>
      tla.store %dst, %source mask %mask {store_dist = #tla.store_dist<pack_b16>} : !ub_f16, !tla.vector<128xf16> mask !tla.mask<128>
    }) {mode = "simd"} : () -> ()
    return
  }
}

// CHECK-LABEL: func.func private @vector_region_
// CHECK: %[[MASK:.*]] = ave.hir.pge <H>
// CHECK-NOT: ave.hir.plt
// CHECK: ave.hir.masked_store <PK_B16> {{.*}}, %[[MASK]],
