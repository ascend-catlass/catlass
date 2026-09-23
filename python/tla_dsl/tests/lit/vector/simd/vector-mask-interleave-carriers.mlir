// RUN: %tla_compile %s --mlir-print-ir-after=tla-vector-region -o %t 2>&1 | %filecheck %s --implicit-check-not=tla.interleave

!data = !tla.tensor<!tla.layout<!tla.shape<64>, !tla.stride<1>, !tla.shape<64>, RowMajor>, !tla.coord<0>, !tla.ptr<f32, ub, 4>>
!bits = !tla.tensor<!tla.layout<!tla.shape<8>, !tla.stride<1>, !tla.shape<8>, RowMajor>, !tla.coord<0>, !tla.ptr<i8, ub, 1>>
!mask = !tla.mask<64>
!vec = !tla.vector<64xf32>

module {
  func.func @mask_interleave_carriers(
      %data_mem: memref<64xf32, #hivm.address_space<ub>>,
      %mask_mem: memref<8xi8, #hivm.address_space<ub>>,
      %count: index, %condition: i1) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c8 = arith.constant 8 : index
    %c64 = arith.constant 64 : index
    %data = tla.tensor_desc %data_mem shape [%c1, %c64, %c1, %c1] stride [%c64, %c1, %c1, %c1] origin_shape [%c1, %c64] coord [%c0, %c0] : memref<64xf32, #hivm.address_space<ub>> -> !data
    %bits = tla.tensor_desc %mask_mem shape [%c1, %c8, %c1, %c1] stride [%c8, %c1, %c1, %c1] origin_shape [%c1, %c8] coord [%c0, %c0] : memref<8xi8, #hivm.address_space<ub>> -> !bits
    "tla.vec.func"() ({
      %value = tla.load %data : !data -> !vec
      %loaded = tla.load %bits : !bits -> !mask
      %tail, %remaining = tla.update_mask %count, f32 : !mask, index
      %cmp = tla.cmp "lt" %value, %value : !vec, !vec -> !mask
      %first, %second = tla.interleave %loaded, %tail : !mask, !mask -> !mask, !mask
      %combined = "tla.bitwise_xor"(%first, %cmp) : (!mask, !mask) -> !mask
      %loop:2 = scf.for %i = %c0 to %count step %c1 iter_args(%a = %combined, %b = %second) -> (!mask, !mask) {
        %selected:2 = scf.if %condition -> (!mask, !mask) {
          %lo, %hi = tla.interleave %a, %b : !mask, !mask -> !mask, !mask
          scf.yield %lo, %hi : !mask, !mask
        } else {
          scf.yield %b, %a : !mask, !mask
        }
        scf.yield %selected#0, %selected#1 : !mask, !mask
      }
      %last0, %last1 = tla.interleave %loop#0, %loop#1 : !mask, !mask -> !mask, !mask
      tla.store %bits, %last0 : !bits, !mask
      tla.store %data, %value mask %last1 : !data, !vec mask !mask
    }) {mode = "simd"} : () -> ()
    return
  }
}

// CHECK-LABEL: func.func private @vector_region_
// CHECK: %[[FIRST:.*]] = "hivm_regbaseintrins.intr.hivm.pintlv.b32"
// CHECK: llvm.extractvalue %[[FIRST]][0]
// CHECK: llvm.extractvalue %[[FIRST]][1]
// CHECK: scf.for {{.*}} -> (vector<256xi1>, vector<256xi1>)
// CHECK: scf.if {{.*}} -> (vector<256xi1>, vector<256xi1>)
// CHECK: %[[NESTED:.*]] = "hivm_regbaseintrins.intr.hivm.pintlv.b32"
// CHECK: %[[LO:.*]] = llvm.extractvalue %[[NESTED]][0]
// CHECK: %[[HI:.*]] = llvm.extractvalue %[[NESTED]][1]
// CHECK: scf.yield %[[LO]], %[[HI]] : vector<256xi1>, vector<256xi1>
// CHECK: %[[LAST:.*]] = "hivm_regbaseintrins.intr.hivm.pintlv.b32"
// CHECK: %[[DST0:.*]] = llvm.extractvalue %[[LAST]][0]
// CHECK: %[[DST1:.*]] = llvm.extractvalue %[[LAST]][1]
// CHECK: %[[STORED:.*]] = builtin.unrealized_conversion_cast %[[DST0]] : vector<256xi1> to vector<64xi1>
// CHECK: ave.hir.masked_store {{.*}}%[[STORED]]
// CHECK: ave.hir.masked_store {{.*}}%[[DST1]]
