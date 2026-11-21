// Bool VectorSSA NORM load/store: IR stays i1; AVE uses same-shape i8 view
// (preserve tile shape/layout) + vector<256xi8>, matching RegTensor<bool> → vector_s8.
// RUN: %tla_compile %s --mlir-print-ir-after=tla-vector-region-simd -o %t 2>&1 | %filecheck %s

!tile = !tla.tensor<!tla.layout<!tla.shape<256>, !tla.stride<1>, !tla.shape<256>, RowMajor>, !tla.coord<0>, !tla.ptr<i1, ub, 1>>

module {
  func.func @bool_load_store(
      %src_memref: memref<256xi1, #hivm.address_space<ub>>,
      %dst_memref: memref<256xi1, #hivm.address_space<ub>>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c256 = arith.constant 256 : index
    %src = tla.tensor_desc %src_memref shape [%c1, %c256, %c1, %c1] stride [%c256, %c1, %c1, %c1] origin_shape [%c1, %c256] coord [%c0, %c0] : memref<256xi1, #hivm.address_space<ub>> -> !tile
    %dst = tla.tensor_desc %dst_memref shape [%c1, %c256, %c1, %c1] stride [%c256, %c1, %c1, %c1] origin_shape [%c1, %c256] coord [%c0, %c0] : memref<256xi1, #hivm.address_space<ub>> -> !tile
    "tla.vector"() ({
      "tla.vec.func"() ({
        %v = tla.load %src : !tile -> !tla.vector<256xi1>
        %m = "tla.create_mask"() {pattern = "ALL", dtype = i1} : () -> !tla.mask<256>
        tla.store %dst, %v mask %m : !tile, !tla.vector<256xi1> mask !tla.mask<256>
      }) {mode = "simd"} : () -> ()
    }) : () -> ()
    return
  }
}

// CHECK-LABEL: func.func private @vector_region_
// Bool load: i1 UB → same-shape i8 view (keeps dynamic offset), then NORM vload.
// CHECK: hivm.hir.pointer_cast
// CHECK-SAME: memref<256xi8, strided<[1], offset: ?>, #hivm.address_space<ub>>
// CHECK: ave.hir.vload <NORM>
// CHECK-SAME: memref<256xi8, strided<[1], offset: ?>, #hivm.address_space<ub>> into vector<256xi8>
// create_mask(dtype=i1) → full predicate (pge ALL).
// CHECK: ave.hir.pge <ALL> : vector<256xi1>
// Bool store: same-shape i8 UB view; NORM_B8 spill of the i8 carrier.
// CHECK: hivm.hir.pointer_cast
// CHECK-SAME: memref<256xi8, strided<[1], offset: ?>, #hivm.address_space<ub>>
// CHECK: ave.hir.masked_store <NORM_B8>
// CHECK-SAME: memref<256xi8, strided<[1], offset: ?>, #hivm.address_space<ub>>, vector<256xi1>, vector<256xi8>
