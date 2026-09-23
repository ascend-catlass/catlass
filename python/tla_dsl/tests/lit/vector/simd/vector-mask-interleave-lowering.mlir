// RUN: %tla_compile %s --mlir-print-ir-after=tla-vector-region -o %t 2>&1 | %filecheck %s --implicit-check-not=tla.interleave --implicit-check-not=ave.hir.interleave

!t8 = !tla.tensor<!tla.layout<!tla.shape<256>, !tla.stride<1>, !tla.shape<256>, RowMajor>, !tla.coord<0>, !tla.ptr<i8, ub, 1>>
!t16 = !tla.tensor<!tla.layout<!tla.shape<128>, !tla.stride<1>, !tla.shape<128>, RowMajor>, !tla.coord<0>, !tla.ptr<f16, ub, 2>>
!t32 = !tla.tensor<!tla.layout<!tla.shape<64>, !tla.stride<1>, !tla.shape<64>, RowMajor>, !tla.coord<0>, !tla.ptr<f32, ub, 4>>

module {
  func.func @mask_interleave_b8(%src: memref<256xi8, #hivm.address_space<ub>>, %dst: memref<256xi8, #hivm.address_space<ub>>) {
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
      %r0, %r1 = tla.interleave %h, %q : !tla.mask<256>, !tla.mask<256> -> !tla.mask<256>, !tla.mask<256>
      %s0 = "tla.where"(%r0, %a, %b) : (!tla.mask<256>, !tla.vector<256xi8>, !tla.vector<256xi8>) -> !tla.vector<256xi8>
      %s1 = "tla.where"(%r1, %b, %a) : (!tla.mask<256>, !tla.vector<256xi8>, !tla.vector<256xi8>) -> !tla.vector<256xi8>
      tla.store %src_t, %s0 mask %r0 : !t8, !tla.vector<256xi8> mask !tla.mask<256>
      tla.store %dst_t, %s1 mask %r1 : !t8, !tla.vector<256xi8> mask !tla.mask<256>
    }) {mode = "simd"} : () -> ()
    return
  }

  func.func @mask_interleave_b16(%src: memref<128xf16, #hivm.address_space<ub>>, %dst: memref<128xf16, #hivm.address_space<ub>>) {
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
      %r0, %r1 = tla.interleave %h, %q : !tla.mask<128>, !tla.mask<128> -> !tla.mask<128>, !tla.mask<128>
      %s0 = "tla.where"(%r0, %a, %b) : (!tla.mask<128>, !tla.vector<128xf16>, !tla.vector<128xf16>) -> !tla.vector<128xf16>
      %s1 = "tla.where"(%r1, %b, %a) : (!tla.mask<128>, !tla.vector<128xf16>, !tla.vector<128xf16>) -> !tla.vector<128xf16>
      tla.store %src_t, %s0 mask %r0 : !t16, !tla.vector<128xf16> mask !tla.mask<128>
      tla.store %dst_t, %s1 mask %r1 : !t16, !tla.vector<128xf16> mask !tla.mask<128>
    }) {mode = "simd"} : () -> ()
    return
  }

  func.func @mask_interleave_b32(%src: memref<64xf32, #hivm.address_space<ub>>, %dst: memref<64xf32, #hivm.address_space<ub>>) {
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
      %r0, %r1 = tla.interleave %h, %q : !tla.mask<64>, !tla.mask<64> -> !tla.mask<64>, !tla.mask<64>
      %s0 = "tla.where"(%r0, %a, %b) : (!tla.mask<64>, !tla.vector<64xf32>, !tla.vector<64xf32>) -> !tla.vector<64xf32>
      %s1 = "tla.where"(%r1, %b, %a) : (!tla.mask<64>, !tla.vector<64xf32>, !tla.vector<64xf32>) -> !tla.vector<64xf32>
      tla.store %src_t, %s0 mask %r0 : !t32, !tla.vector<64xf32> mask !tla.mask<64>
      tla.store %dst_t, %s1 mask %r1 : !t32, !tla.vector<64xf32> mask !tla.mask<64>
    }) {mode = "simd"} : () -> ()
    return
  }

}

// CHECK-LABEL: func.func @mask_interleave_b8
// CHECK-LABEL: func.func private @vector_region_
// CHECK-NOT: call
// CHECK: %[[PAIR:.*]] = "hivm_regbaseintrins.intr.hivm.pintlv.b8"
// CHECK-SAME: (vector<256xi1>, vector<256xi1>) -> !llvm.struct<(vector<256xi1>, vector<256xi1>)>
// CHECK-NEXT: %[[LO:.*]] = llvm.extractvalue %[[PAIR]][0] : !llvm.struct<(vector<256xi1>, vector<256xi1>)>
// CHECK-NEXT: %[[HI:.*]] = llvm.extractvalue %[[PAIR]][1] : !llvm.struct<(vector<256xi1>, vector<256xi1>)>
// CHECK: ave.hir.vsel %[[LO]]
// CHECK: ave.hir.vsel %[[HI]]
// CHECK: ave.hir.masked_store {{.*}}%[[LO]]
// CHECK: ave.hir.masked_store {{.*}}%[[HI]]
// CHECK-NOT: call
// CHECK: return

// CHECK-LABEL: func.func @mask_interleave_b16
// CHECK-LABEL: func.func private @vector_region_
// CHECK-NOT: call
// CHECK: %[[PAIR:.*]] = "hivm_regbaseintrins.intr.hivm.pintlv.b16"
// CHECK-SAME: (vector<256xi1>, vector<256xi1>) -> !llvm.struct<(vector<256xi1>, vector<256xi1>)>
// CHECK-NEXT: %[[LO:.*]] = llvm.extractvalue %[[PAIR]][0] : !llvm.struct<(vector<256xi1>, vector<256xi1>)>
// CHECK-NEXT: %[[HI:.*]] = llvm.extractvalue %[[PAIR]][1] : !llvm.struct<(vector<256xi1>, vector<256xi1>)>
// CHECK: ave.hir.vsel %[[LO]]
// CHECK: ave.hir.vsel %[[HI]]
// CHECK: ave.hir.masked_store {{.*}}%[[LO]]
// CHECK: ave.hir.masked_store {{.*}}%[[HI]]
// CHECK-NOT: call
// CHECK: return

// CHECK-LABEL: func.func @mask_interleave_b32
// CHECK-LABEL: func.func private @vector_region_
// CHECK-NOT: call
// CHECK: %[[PAIR:.*]] = "hivm_regbaseintrins.intr.hivm.pintlv.b32"
// CHECK-SAME: (vector<256xi1>, vector<256xi1>) -> !llvm.struct<(vector<256xi1>, vector<256xi1>)>
// CHECK-NEXT: %[[LO:.*]] = llvm.extractvalue %[[PAIR]][0] : !llvm.struct<(vector<256xi1>, vector<256xi1>)>
// CHECK-NEXT: %[[HI:.*]] = llvm.extractvalue %[[PAIR]][1] : !llvm.struct<(vector<256xi1>, vector<256xi1>)>
// CHECK: ave.hir.vsel %[[LO]]
// CHECK: ave.hir.vsel %[[HI]]
// CHECK: ave.hir.masked_store {{.*}}%[[LO]]
// CHECK: ave.hir.masked_store {{.*}}%[[HI]]
// CHECK-NOT: call
// CHECK: return
