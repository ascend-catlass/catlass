// RUN: %tla_compile %s --mlir-print-ir-after=tla-vector-region -o %t 2>&1 | %filecheck %s --implicit-check-not=tla.deinterleave --implicit-check-not=ave.hir.vdintlv

!t8 = !tla.tensor<!tla.layout<!tla.shape<256>, !tla.stride<1>, !tla.shape<256>, RowMajor>, !tla.coord<0>, !tla.ptr<i8, ub, 1>>
!t16 = !tla.tensor<!tla.layout<!tla.shape<128>, !tla.stride<1>, !tla.shape<128>, RowMajor>, !tla.coord<0>, !tla.ptr<f16, ub, 2>>
!t32 = !tla.tensor<!tla.layout<!tla.shape<64>, !tla.stride<1>, !tla.shape<64>, RowMajor>, !tla.coord<0>, !tla.ptr<f32, ub, 4>>

!bits = !tla.tensor<!tla.layout<!tla.shape<8>, !tla.stride<1>, !tla.shape<8>, RowMajor>, !tla.coord<0>, !tla.ptr<i8, ub, 1>>
!mask = !tla.mask<64>
!vec = !tla.vector<64xf32>

module {
  func.func @mask_deinterleave_b8(%src: memref<256xi8, #hivm.address_space<ub>>, %dst: memref<256xi8, #hivm.address_space<ub>>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %n = arith.constant 256 : index
    %src_t = tla.tensor_desc %src shape [%c1, %n, %c1, %c1] stride [%n, %c1, %c1, %c1] origin_shape [%c1, %n] coord [%c0, %c0] : memref<256xi8, #hivm.address_space<ub>> -> !t8
    %dst_t = tla.tensor_desc %dst shape [%c1, %n, %c1, %c1] stride [%n, %c1, %c1, %c1] origin_shape [%c1, %n] coord [%c0, %c0] : memref<256xi8, #hivm.address_space<ub>> -> !t8
    "tla.vec.func"() ({
      %a = tla.load %src_t : !t8 -> !tla.vector<256xi8>
      %b = tla.load %dst_t : !t8 -> !tla.vector<256xi8>
      %h = "tla.create_mask"() {pattern = "H", dtype = i8} : () -> !tla.mask<256>
      %q = "tla.create_mask"() {pattern = "Q", dtype = i8} : () -> !tla.mask<256>
      %r0, %r1 = tla.deinterleave %h, %q : !tla.mask<256>, !tla.mask<256> -> !tla.mask<256>, !tla.mask<256>
      %s0 = "tla.where"(%r0, %a, %b) : (!tla.mask<256>, !tla.vector<256xi8>, !tla.vector<256xi8>) -> !tla.vector<256xi8>
      %s1 = "tla.where"(%r1, %b, %a) : (!tla.mask<256>, !tla.vector<256xi8>, !tla.vector<256xi8>) -> !tla.vector<256xi8>
      tla.store %src_t, %s0 mask %r0 : !t8, !tla.vector<256xi8> mask !tla.mask<256>
      tla.store %dst_t, %s1 mask %r1 : !t8, !tla.vector<256xi8> mask !tla.mask<256>
    }) {mode = "simd"} : () -> ()
    return
  }

  func.func @mask_deinterleave_b16(%src: memref<128xf16, #hivm.address_space<ub>>, %dst: memref<128xf16, #hivm.address_space<ub>>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %n = arith.constant 128 : index
    %src_t = tla.tensor_desc %src shape [%c1, %n, %c1, %c1] stride [%n, %c1, %c1, %c1] origin_shape [%c1, %n] coord [%c0, %c0] : memref<128xf16, #hivm.address_space<ub>> -> !t16
    %dst_t = tla.tensor_desc %dst shape [%c1, %n, %c1, %c1] stride [%n, %c1, %c1, %c1] origin_shape [%c1, %n] coord [%c0, %c0] : memref<128xf16, #hivm.address_space<ub>> -> !t16
    "tla.vec.func"() ({
      %a = tla.load %src_t : !t16 -> !tla.vector<128xf16>
      %b = tla.load %dst_t : !t16 -> !tla.vector<128xf16>
      %h = "tla.create_mask"() {pattern = "H", dtype = f16} : () -> !tla.mask<128>
      %q = "tla.create_mask"() {pattern = "Q", dtype = f16} : () -> !tla.mask<128>
      %r0, %r1 = tla.deinterleave %h, %q : !tla.mask<128>, !tla.mask<128> -> !tla.mask<128>, !tla.mask<128>
      %s0 = "tla.where"(%r0, %a, %b) : (!tla.mask<128>, !tla.vector<128xf16>, !tla.vector<128xf16>) -> !tla.vector<128xf16>
      %s1 = "tla.where"(%r1, %b, %a) : (!tla.mask<128>, !tla.vector<128xf16>, !tla.vector<128xf16>) -> !tla.vector<128xf16>
      tla.store %src_t, %s0 mask %r0 : !t16, !tla.vector<128xf16> mask !tla.mask<128>
      tla.store %dst_t, %s1 mask %r1 : !t16, !tla.vector<128xf16> mask !tla.mask<128>
    }) {mode = "simd"} : () -> ()
    return
  }

  func.func @mask_deinterleave_b32(%src: memref<64xf32, #hivm.address_space<ub>>, %dst: memref<64xf32, #hivm.address_space<ub>>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %n = arith.constant 64 : index
    %src_t = tla.tensor_desc %src shape [%c1, %n, %c1, %c1] stride [%n, %c1, %c1, %c1] origin_shape [%c1, %n] coord [%c0, %c0] : memref<64xf32, #hivm.address_space<ub>> -> !t32
    %dst_t = tla.tensor_desc %dst shape [%c1, %n, %c1, %c1] stride [%n, %c1, %c1, %c1] origin_shape [%c1, %n] coord [%c0, %c0] : memref<64xf32, #hivm.address_space<ub>> -> !t32
    "tla.vec.func"() ({
      %a = tla.load %src_t : !t32 -> !tla.vector<64xf32>
      %b = tla.load %dst_t : !t32 -> !tla.vector<64xf32>
      %h = "tla.create_mask"() {pattern = "H", dtype = f32} : () -> !tla.mask<64>
      %q = "tla.create_mask"() {pattern = "Q", dtype = f32} : () -> !tla.mask<64>
      %r0, %r1 = tla.deinterleave %h, %q : !tla.mask<64>, !tla.mask<64> -> !tla.mask<64>, !tla.mask<64>
      %s0 = "tla.where"(%r0, %a, %b) : (!tla.mask<64>, !tla.vector<64xf32>, !tla.vector<64xf32>) -> !tla.vector<64xf32>
      %s1 = "tla.where"(%r1, %b, %a) : (!tla.mask<64>, !tla.vector<64xf32>, !tla.vector<64xf32>) -> !tla.vector<64xf32>
      tla.store %src_t, %s0 mask %r0 : !t32, !tla.vector<64xf32> mask !tla.mask<64>
      tla.store %dst_t, %s1 mask %r1 : !t32, !tla.vector<64xf32> mask !tla.mask<64>
    }) {mode = "simd"} : () -> ()
    return
  }

  func.func @mask_deinterleave_carriers(
      %data_mem: memref<64xf32, #hivm.address_space<ub>>,
      %mask_mem: memref<8xi8, #hivm.address_space<ub>>,
      %count: index, %condition: i1) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c8 = arith.constant 8 : index
    %c64 = arith.constant 64 : index
    %data = tla.tensor_desc %data_mem shape [%c1, %c64, %c1, %c1] stride [%c64, %c1, %c1, %c1] origin_shape [%c1, %c64] coord [%c0, %c0] : memref<64xf32, #hivm.address_space<ub>> -> !t32
    %bits = tla.tensor_desc %mask_mem shape [%c1, %c8, %c1, %c1] stride [%c8, %c1, %c1, %c1] origin_shape [%c1, %c8] coord [%c0, %c0] : memref<8xi8, #hivm.address_space<ub>> -> !bits
    "tla.vec.func"() ({
      %value = tla.load %data : !t32 -> !vec
      %loaded = tla.load %bits : !bits -> !mask
      %tail, %remaining = tla.update_mask %count, f32 : !mask, index
      %cmp = tla.cmp "lt" %value, %value : !vec, !vec -> !mask
      %first, %second = tla.deinterleave %loaded, %tail : !mask, !mask -> !mask, !mask
      %combined = "tla.bitwise_xor"(%first, %cmp) : (!mask, !mask) -> !mask
      %loop:2 = scf.for %i = %c0 to %count step %c1 iter_args(%a = %combined, %b = %second) -> (!mask, !mask) {
        %selected:2 = scf.if %condition -> (!mask, !mask) {
          %even, %odd = tla.deinterleave %a, %b : !mask, !mask -> !mask, !mask
          scf.yield %even, %odd : !mask, !mask
        } else {
          scf.yield %b, %a : !mask, !mask
        }
        scf.yield %selected#0, %selected#1 : !mask, !mask
      }
      %last0, %last1 = tla.deinterleave %loop#0, %loop#1 : !mask, !mask -> !mask, !mask
      tla.store %bits, %last0 : !bits, !mask
      tla.store %data, %value mask %last1 : !t32, !vec mask !mask
    }) {mode = "simd"} : () -> ()
    return
  }
}

// CHECK-LABEL: func.func @mask_deinterleave_b8
// CHECK-LABEL: func.func private @vector_region_
// CHECK-NOT: call
// CHECK: %[[PAIR:.*]] = "hivm_regbaseintrins.intr.hivm.pdintlv.b8"
// CHECK-SAME: (vector<256xi1>, vector<256xi1>) -> !llvm.struct<(vector<256xi1>, vector<256xi1>)>
// CHECK-NEXT: %[[EVEN:.*]] = llvm.extractvalue %[[PAIR]][0] : !llvm.struct<(vector<256xi1>, vector<256xi1>)>
// CHECK-NEXT: %[[ODD:.*]] = llvm.extractvalue %[[PAIR]][1] : !llvm.struct<(vector<256xi1>, vector<256xi1>)>
// CHECK: ave.hir.vsel %[[EVEN]]
// CHECK: ave.hir.vsel %[[ODD]]
// CHECK: ave.hir.masked_store {{.*}}%[[EVEN]]
// CHECK: ave.hir.masked_store {{.*}}%[[ODD]]
// CHECK-NOT: call
// CHECK: return

// CHECK-LABEL: func.func @mask_deinterleave_b16
// CHECK-LABEL: func.func private @vector_region_
// CHECK-NOT: call
// CHECK: %[[PAIR:.*]] = "hivm_regbaseintrins.intr.hivm.pdintlv.b16"
// CHECK-SAME: (vector<256xi1>, vector<256xi1>) -> !llvm.struct<(vector<256xi1>, vector<256xi1>)>
// CHECK-NEXT: %[[EVEN:.*]] = llvm.extractvalue %[[PAIR]][0] : !llvm.struct<(vector<256xi1>, vector<256xi1>)>
// CHECK-NEXT: %[[ODD:.*]] = llvm.extractvalue %[[PAIR]][1] : !llvm.struct<(vector<256xi1>, vector<256xi1>)>
// CHECK: ave.hir.vsel %[[EVEN]]
// CHECK: ave.hir.vsel %[[ODD]]
// CHECK: ave.hir.masked_store {{.*}}%[[EVEN]]
// CHECK: ave.hir.masked_store {{.*}}%[[ODD]]
// CHECK-NOT: call
// CHECK: return

// CHECK-LABEL: func.func @mask_deinterleave_b32
// CHECK-LABEL: func.func private @vector_region_
// CHECK-NOT: call
// CHECK: %[[PAIR:.*]] = "hivm_regbaseintrins.intr.hivm.pdintlv.b32"
// CHECK-SAME: (vector<256xi1>, vector<256xi1>) -> !llvm.struct<(vector<256xi1>, vector<256xi1>)>
// CHECK-NEXT: %[[EVEN:.*]] = llvm.extractvalue %[[PAIR]][0] : !llvm.struct<(vector<256xi1>, vector<256xi1>)>
// CHECK-NEXT: %[[ODD:.*]] = llvm.extractvalue %[[PAIR]][1] : !llvm.struct<(vector<256xi1>, vector<256xi1>)>
// CHECK: ave.hir.vsel %[[EVEN]]
// CHECK: ave.hir.vsel %[[ODD]]
// CHECK: ave.hir.masked_store {{.*}}%[[EVEN]]
// CHECK: ave.hir.masked_store {{.*}}%[[ODD]]
// CHECK-NOT: call
// CHECK: return

// Both SCF results keep mask<64> semantics despite vector<256xi1> carriers.
// CHECK-LABEL: func.func @mask_deinterleave_carriers
// CHECK-LABEL: func.func private @vector_region_
// CHECK: %[[FIRST:.*]] = "hivm_regbaseintrins.intr.hivm.pdintlv.b32"
// CHECK: llvm.extractvalue %[[FIRST]][0]
// CHECK: llvm.extractvalue %[[FIRST]][1]
// CHECK: scf.for {{.*}} -> (vector<256xi1>, vector<256xi1>)
// CHECK: scf.if {{.*}} -> (vector<256xi1>, vector<256xi1>)
// CHECK: %[[NESTED:.*]] = "hivm_regbaseintrins.intr.hivm.pdintlv.b32"
// CHECK: %[[EVEN:.*]] = llvm.extractvalue %[[NESTED]][0]
// CHECK: %[[ODD:.*]] = llvm.extractvalue %[[NESTED]][1]
// CHECK: scf.yield %[[EVEN]], %[[ODD]] : vector<256xi1>, vector<256xi1>
// CHECK: %[[LAST:.*]] = "hivm_regbaseintrins.intr.hivm.pdintlv.b32"
// CHECK: %[[DST0:.*]] = llvm.extractvalue %[[LAST]][0]
// CHECK: %[[DST1:.*]] = llvm.extractvalue %[[LAST]][1]
// CHECK: %[[STORED:.*]] = builtin.unrealized_conversion_cast %[[DST0]] : vector<256xi1> to vector<64xi1>
// CHECK: ave.hir.masked_store {{.*}}%[[STORED]]
// CHECK: ave.hir.masked_store {{.*}}%[[DST1]]
