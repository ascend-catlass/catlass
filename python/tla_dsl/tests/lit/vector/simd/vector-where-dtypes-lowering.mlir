// RUN: %tla_compile %s --mlir-print-ir-after=tla-vector-region-simd -o %t 2>&1 | %filecheck %s --implicit-check-not=tla.where

!bf16vec = !tla.tensor<!tla.layout<!tla.shape<128>, !tla.stride<1>, !tla.shape<128>, RowMajor>, !tla.coord<0>, !tla.ptr<bf16, ub, 2>>
!i8vec = !tla.tensor<!tla.layout<!tla.shape<256>, !tla.stride<1>, !tla.shape<256>, RowMajor>, !tla.coord<0>, !tla.ptr<i8, ub, 1>>
!i64vec = !tla.tensor<!tla.layout<!tla.shape<32>, !tla.stride<1>, !tla.shape<32>, RowMajor>, !tla.coord<0>, !tla.ptr<i64, ub, 8>>
!u64vec = !tla.tensor<!tla.layout<!tla.shape<32>, !tla.stride<1>, !tla.shape<32>, RowMajor>, !tla.coord<0>, !tla.ptr<ui64, ub, 8>>
!boolvec = !tla.tensor<!tla.layout<!tla.shape<256>, !tla.stride<1>, !tla.shape<256>, RowMajor>, !tla.coord<0>, !tla.ptr<i1, ub, 1>>
!f8e4vec = !tla.tensor<!tla.layout<!tla.shape<256>, !tla.stride<1>, !tla.shape<256>, RowMajor>, !tla.coord<0>, !tla.ptr<f8E4M3FN, ub, 1>>
!f8e5vec = !tla.tensor<!tla.layout<!tla.shape<256>, !tla.stride<1>, !tla.shape<256>, RowMajor>, !tla.coord<0>, !tla.ptr<f8E5M2, ub, 1>>
!hif8vec = !tla.tensor<!tla.layout<!tla.shape<256>, !tla.stride<1>, !tla.shape<256>, RowMajor>, !tla.coord<0>, !tla.ptr<!tla.hif8, ub, 1>>
!f4e2vec = !tla.tensor<!tla.layout<!tla.shape<256>, !tla.stride<1>, !tla.shape<256>, RowMajor>, !tla.coord<0>, !tla.ptr<!tla.f4e2m1x2, ub, 32>>
!f4e1vec = !tla.tensor<!tla.layout<!tla.shape<256>, !tla.stride<1>, !tla.shape<256>, RowMajor>, !tla.coord<0>, !tla.ptr<!tla.f4e1m2x2, ub, 32>>

module {
  func.func @where_bf16(
      %x_memref: memref<128xbf16, #hivm.address_space<ub>>,
      %y_memref: memref<128xbf16, #hivm.address_space<ub>>,
      %dst_memref: memref<128xbf16, #hivm.address_space<ub>>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c128 = arith.constant 128 : index
    %x = tla.tensor_desc %x_memref shape [%c1, %c128, %c1, %c1] stride [%c128, %c1, %c1, %c1] origin_shape [%c1, %c128] coord [%c0, %c0] : memref<128xbf16, #hivm.address_space<ub>> -> !bf16vec
    %y = tla.tensor_desc %y_memref shape [%c1, %c128, %c1, %c1] stride [%c128, %c1, %c1, %c1] origin_shape [%c1, %c128] coord [%c0, %c0] : memref<128xbf16, #hivm.address_space<ub>> -> !bf16vec
    %dst = tla.tensor_desc %dst_memref shape [%c1, %c128, %c1, %c1] stride [%c128, %c1, %c1, %c1] origin_shape [%c1, %c128] coord [%c0, %c0] : memref<128xbf16, #hivm.address_space<ub>> -> !bf16vec
    "tla.vec.func"() ({
      %xv = tla.load %x : !bf16vec -> !tla.vector<128xbf16>
      %yv = tla.load %y : !bf16vec -> !tla.vector<128xbf16>
      %mask = "tla.create_mask"() {pattern = "H", dtype = bf16} : () -> !tla.mask<128>
      %selected = "tla.where"(%mask, %xv, %yv) : (!tla.mask<128>, !tla.vector<128xbf16>, !tla.vector<128xbf16>) -> !tla.vector<128xbf16>
      tla.store %dst, %selected : !bf16vec, !tla.vector<128xbf16>
    }) {mode = "simd"} : () -> ()
    return
  }

  func.func @where_i64(
      %x_memref: memref<32xi64, #hivm.address_space<ub>>,
      %y_memref: memref<32xi64, #hivm.address_space<ub>>,
      %dst_memref: memref<32xi64, #hivm.address_space<ub>>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c32 = arith.constant 32 : index
    %x = tla.tensor_desc %x_memref shape [%c1, %c32, %c1, %c1] stride [%c32, %c1, %c1, %c1] origin_shape [%c1, %c32] coord [%c0, %c0] : memref<32xi64, #hivm.address_space<ub>> -> !i64vec
    %y = tla.tensor_desc %y_memref shape [%c1, %c32, %c1, %c1] stride [%c32, %c1, %c1, %c1] origin_shape [%c1, %c32] coord [%c0, %c0] : memref<32xi64, #hivm.address_space<ub>> -> !i64vec
    %dst = tla.tensor_desc %dst_memref shape [%c1, %c32, %c1, %c1] stride [%c32, %c1, %c1, %c1] origin_shape [%c1, %c32] coord [%c0, %c0] : memref<32xi64, #hivm.address_space<ub>> -> !i64vec
    "tla.vec.func"() ({
      %xv = tla.load %x : !i64vec -> !tla.vector<32xi64>
      %yv = tla.load %y : !i64vec -> !tla.vector<32xi64>
      %mask = "tla.create_mask"() {pattern = "VL16", dtype = i64} : () -> !tla.mask<32>
      %selected = "tla.where"(%mask, %xv, %yv) : (!tla.mask<32>, !tla.vector<32xi64>, !tla.vector<32xi64>) -> !tla.vector<32xi64>
      tla.store %dst, %selected : !i64vec, !tla.vector<32xi64>
    }) {mode = "simd"} : () -> ()
    return
  }

  func.func @where_u64(
      %x_memref: memref<32xui64, #hivm.address_space<ub>>,
      %y_memref: memref<32xui64, #hivm.address_space<ub>>,
      %dst_memref: memref<32xui64, #hivm.address_space<ub>>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c32 = arith.constant 32 : index
    %x = tla.tensor_desc %x_memref shape [%c1, %c32, %c1, %c1] stride [%c32, %c1, %c1, %c1] origin_shape [%c1, %c32] coord [%c0, %c0] : memref<32xui64, #hivm.address_space<ub>> -> !u64vec
    %y = tla.tensor_desc %y_memref shape [%c1, %c32, %c1, %c1] stride [%c32, %c1, %c1, %c1] origin_shape [%c1, %c32] coord [%c0, %c0] : memref<32xui64, #hivm.address_space<ub>> -> !u64vec
    %dst = tla.tensor_desc %dst_memref shape [%c1, %c32, %c1, %c1] stride [%c32, %c1, %c1, %c1] origin_shape [%c1, %c32] coord [%c0, %c0] : memref<32xui64, #hivm.address_space<ub>> -> !u64vec
    "tla.vec.func"() ({
      %xv = tla.load %x : !u64vec -> !tla.vector<32xui64>
      %yv = tla.load %y : !u64vec -> !tla.vector<32xui64>
      %mask = "tla.create_mask"() {pattern = "VL16", dtype = ui64} : () -> !tla.mask<32>
      %selected = "tla.where"(%mask, %xv, %yv) : (!tla.mask<32>, !tla.vector<32xui64>, !tla.vector<32xui64>) -> !tla.vector<32xui64>
      tla.store %dst, %selected : !u64vec, !tla.vector<32xui64>
    }) {mode = "simd"} : () -> ()
    return
  }

  func.func @where_bool(
      %x_memref: memref<256xi1, #hivm.address_space<ub>>,
      %y_memref: memref<256xi1, #hivm.address_space<ub>>,
      %dst_memref: memref<256xi1, #hivm.address_space<ub>>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c256 = arith.constant 256 : index
    %x = tla.tensor_desc %x_memref shape [%c1, %c256, %c1, %c1] stride [%c256, %c1, %c1, %c1] origin_shape [%c1, %c256] coord [%c0, %c0] : memref<256xi1, #hivm.address_space<ub>> -> !boolvec
    %y = tla.tensor_desc %y_memref shape [%c1, %c256, %c1, %c1] stride [%c256, %c1, %c1, %c1] origin_shape [%c1, %c256] coord [%c0, %c0] : memref<256xi1, #hivm.address_space<ub>> -> !boolvec
    %dst = tla.tensor_desc %dst_memref shape [%c1, %c256, %c1, %c1] stride [%c256, %c1, %c1, %c1] origin_shape [%c1, %c256] coord [%c0, %c0] : memref<256xi1, #hivm.address_space<ub>> -> !boolvec
    "tla.vec.func"() ({
      %xv = tla.load %x : !boolvec -> !tla.vector<256xi1>
      %yv = tla.load %y : !boolvec -> !tla.vector<256xi1>
      %mask = "tla.create_mask"() {pattern = "H", dtype = i1} : () -> !tla.mask<256>
      %selected = "tla.where"(%mask, %xv, %yv) : (!tla.mask<256>, !tla.vector<256xi1>, !tla.vector<256xi1>) -> !tla.vector<256xi1>
      tla.store %dst, %selected : !boolvec, !tla.vector<256xi1>
    }) {mode = "simd"} : () -> ()
    return
  }
  func.func @where_f8e4m3fn(
      %x_memref: memref<256xf8E4M3FN, #hivm.address_space<ub>>,
      %y_memref: memref<256xf8E4M3FN, #hivm.address_space<ub>>,
      %dst_memref: memref<256xf8E4M3FN, #hivm.address_space<ub>>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c256 = arith.constant 256 : index
    %x = tla.tensor_desc %x_memref shape [%c1, %c256, %c1, %c1] stride [%c256, %c1, %c1, %c1] origin_shape [%c1, %c256] coord [%c0, %c0] : memref<256xf8E4M3FN, #hivm.address_space<ub>> -> !f8e4vec
    %y = tla.tensor_desc %y_memref shape [%c1, %c256, %c1, %c1] stride [%c256, %c1, %c1, %c1] origin_shape [%c1, %c256] coord [%c0, %c0] : memref<256xf8E4M3FN, #hivm.address_space<ub>> -> !f8e4vec
    %dst = tla.tensor_desc %dst_memref shape [%c1, %c256, %c1, %c1] stride [%c256, %c1, %c1, %c1] origin_shape [%c1, %c256] coord [%c0, %c0] : memref<256xf8E4M3FN, #hivm.address_space<ub>> -> !f8e4vec
    "tla.vec.func"() ({
      %xv = tla.load %x : !f8e4vec -> !tla.vector<256xf8E4M3FN>
      %yv = tla.load %y : !f8e4vec -> !tla.vector<256xf8E4M3FN>
      %mask = "tla.create_mask"() {pattern = "H", dtype = f8E4M3FN} : () -> !tla.mask<256>
      %selected = "tla.where"(%mask, %xv, %yv) : (!tla.mask<256>, !tla.vector<256xf8E4M3FN>, !tla.vector<256xf8E4M3FN>) -> !tla.vector<256xf8E4M3FN>
      tla.store %dst, %selected : !f8e4vec, !tla.vector<256xf8E4M3FN>
    }) {mode = "simd"} : () -> ()
    return
  }

  func.func @where_f8e5m2(
      %x_memref: memref<256xf8E5M2, #hivm.address_space<ub>>,
      %y_memref: memref<256xf8E5M2, #hivm.address_space<ub>>,
      %dst_memref: memref<256xf8E5M2, #hivm.address_space<ub>>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c256 = arith.constant 256 : index
    %x = tla.tensor_desc %x_memref shape [%c1, %c256, %c1, %c1] stride [%c256, %c1, %c1, %c1] origin_shape [%c1, %c256] coord [%c0, %c0] : memref<256xf8E5M2, #hivm.address_space<ub>> -> !f8e5vec
    %y = tla.tensor_desc %y_memref shape [%c1, %c256, %c1, %c1] stride [%c256, %c1, %c1, %c1] origin_shape [%c1, %c256] coord [%c0, %c0] : memref<256xf8E5M2, #hivm.address_space<ub>> -> !f8e5vec
    %dst = tla.tensor_desc %dst_memref shape [%c1, %c256, %c1, %c1] stride [%c256, %c1, %c1, %c1] origin_shape [%c1, %c256] coord [%c0, %c0] : memref<256xf8E5M2, #hivm.address_space<ub>> -> !f8e5vec
    "tla.vec.func"() ({
      %xv = tla.load %x : !f8e5vec -> !tla.vector<256xf8E5M2>
      %yv = tla.load %y : !f8e5vec -> !tla.vector<256xf8E5M2>
      %mask = "tla.create_mask"() {pattern = "H", dtype = f8E5M2} : () -> !tla.mask<256>
      %selected = "tla.where"(%mask, %xv, %yv) : (!tla.mask<256>, !tla.vector<256xf8E5M2>, !tla.vector<256xf8E5M2>) -> !tla.vector<256xf8E5M2>
      tla.store %dst, %selected : !f8e5vec, !tla.vector<256xf8E5M2>
    }) {mode = "simd"} : () -> ()
    return
  }

  func.func @where_hif8(
      %x_memref: memref<256xi8, #hivm.address_space<ub>>,
      %y_memref: memref<256xi8, #hivm.address_space<ub>>,
      %dst_memref: memref<256xi8, #hivm.address_space<ub>>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c256 = arith.constant 256 : index
    %x = tla.tensor_desc %x_memref shape [%c1, %c256, %c1, %c1] stride [%c256, %c1, %c1, %c1] origin_shape [%c1, %c256] coord [%c0, %c0] : memref<256xi8, #hivm.address_space<ub>> -> !hif8vec
    %y = tla.tensor_desc %y_memref shape [%c1, %c256, %c1, %c1] stride [%c256, %c1, %c1, %c1] origin_shape [%c1, %c256] coord [%c0, %c0] : memref<256xi8, #hivm.address_space<ub>> -> !hif8vec
    %dst = tla.tensor_desc %dst_memref shape [%c1, %c256, %c1, %c1] stride [%c256, %c1, %c1, %c1] origin_shape [%c1, %c256] coord [%c0, %c0] : memref<256xi8, #hivm.address_space<ub>> -> !hif8vec
    "tla.vec.func"() ({
      %xv = tla.load %x : !hif8vec -> !tla.vector<256x!tla.hif8>
      %yv = tla.load %y : !hif8vec -> !tla.vector<256x!tla.hif8>
      %mask = "tla.create_mask"() {pattern = "H", dtype = i8} : () -> !tla.mask<256>
      %selected = "tla.where"(%mask, %xv, %yv) : (!tla.mask<256>, !tla.vector<256x!tla.hif8>, !tla.vector<256x!tla.hif8>) -> !tla.vector<256x!tla.hif8>
      tla.store %dst, %selected : !hif8vec, !tla.vector<256x!tla.hif8>
    }) {mode = "simd"} : () -> ()
    return
  }

  func.func @where_f4e2m1x2(
      %x_memref: memref<256xi8, #hivm.address_space<ub>>,
      %y_memref: memref<256xi8, #hivm.address_space<ub>>,
      %dst_memref: memref<256xi8, #hivm.address_space<ub>>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c256 = arith.constant 256 : index
    %x = tla.tensor_desc %x_memref shape [%c1, %c256, %c1, %c1] stride [%c256, %c1, %c1, %c1] origin_shape [%c1, %c256] coord [%c0, %c0] : memref<256xi8, #hivm.address_space<ub>> -> !f4e2vec
    %y = tla.tensor_desc %y_memref shape [%c1, %c256, %c1, %c1] stride [%c256, %c1, %c1, %c1] origin_shape [%c1, %c256] coord [%c0, %c0] : memref<256xi8, #hivm.address_space<ub>> -> !f4e2vec
    %dst = tla.tensor_desc %dst_memref shape [%c1, %c256, %c1, %c1] stride [%c256, %c1, %c1, %c1] origin_shape [%c1, %c256] coord [%c0, %c0] : memref<256xi8, #hivm.address_space<ub>> -> !f4e2vec
    "tla.vec.func"() ({
      %xv = tla.load %x : !f4e2vec -> !tla.vector<256x!tla.f4e2m1x2>
      %yv = tla.load %y : !f4e2vec -> !tla.vector<256x!tla.f4e2m1x2>
      %mask = "tla.create_mask"() {pattern = "H", dtype = i8} : () -> !tla.mask<256>
      %selected = "tla.where"(%mask, %xv, %yv) : (!tla.mask<256>, !tla.vector<256x!tla.f4e2m1x2>, !tla.vector<256x!tla.f4e2m1x2>) -> !tla.vector<256x!tla.f4e2m1x2>
      tla.store %dst, %selected : !f4e2vec, !tla.vector<256x!tla.f4e2m1x2>
    }) {mode = "simd"} : () -> ()
    return
  }

  func.func @where_f4e1m2x2(
      %x_memref: memref<256xi8, #hivm.address_space<ub>>,
      %y_memref: memref<256xi8, #hivm.address_space<ub>>,
      %dst_memref: memref<256xi8, #hivm.address_space<ub>>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c256 = arith.constant 256 : index
    %x = tla.tensor_desc %x_memref shape [%c1, %c256, %c1, %c1] stride [%c256, %c1, %c1, %c1] origin_shape [%c1, %c256] coord [%c0, %c0] : memref<256xi8, #hivm.address_space<ub>> -> !f4e1vec
    %y = tla.tensor_desc %y_memref shape [%c1, %c256, %c1, %c1] stride [%c256, %c1, %c1, %c1] origin_shape [%c1, %c256] coord [%c0, %c0] : memref<256xi8, #hivm.address_space<ub>> -> !f4e1vec
    %dst = tla.tensor_desc %dst_memref shape [%c1, %c256, %c1, %c1] stride [%c256, %c1, %c1, %c1] origin_shape [%c1, %c256] coord [%c0, %c0] : memref<256xi8, #hivm.address_space<ub>> -> !f4e1vec
    "tla.vec.func"() ({
      %xv = tla.load %x : !f4e1vec -> !tla.vector<256x!tla.f4e1m2x2>
      %yv = tla.load %y : !f4e1vec -> !tla.vector<256x!tla.f4e1m2x2>
      %mask = "tla.create_mask"() {pattern = "H", dtype = i8} : () -> !tla.mask<256>
      %selected = "tla.where"(%mask, %xv, %yv) : (!tla.mask<256>, !tla.vector<256x!tla.f4e1m2x2>, !tla.vector<256x!tla.f4e1m2x2>) -> !tla.vector<256x!tla.f4e1m2x2>
      tla.store %dst, %selected : !f4e1vec, !tla.vector<256x!tla.f4e1m2x2>
    }) {mode = "simd"} : () -> ()
    return
  }

}

// CHECK-LABEL: func.func private @vector_region_
// CHECK: ave.hir.vsel {{.*}} : vector<256xi1>, vector<128xbf16>
// CHECK-LABEL: func.func private @vector_region_
// CHECK: call @select_int64_t({{.*}}) : (vector<256xi1>, vector<32xi64>, vector<32xi64>) -> vector<32xi64>
// CHECK-LABEL: func.func private @vector_region_
// CHECK: call @select_uint64_t({{.*}}) : (vector<256xi1>, vector<32xi64>, vector<32xi64>) -> vector<32xi64>
// CHECK-LABEL: func.func private @vector_region_
// CHECK: ave.hir.vsel {{.*}} : vector<256xi1>, vector<256xi8>

// CHECK-LABEL: func.func private @vector_region_
// CHECK: ave.hir.vsel {{.*}} : vector<256xi1>, vector<256xf8E4M3FN>
// CHECK-LABEL: func.func private @vector_region_
// CHECK: ave.hir.vsel {{.*}} : vector<256xi1>, vector<256xf8E5M2>
// CHECK-LABEL: func.func private @vector_region_
// CHECK: ave.hir.vsel {{.*}} : vector<256xi1>, vector<256xi8>
// CHECK-LABEL: func.func private @vector_region_
// CHECK: ave.hir.vsel {{.*}} : vector<256xi1>, vector<256xi8>
// CHECK-LABEL: func.func private @vector_region_
// CHECK: ave.hir.vsel {{.*}} : vector<256xi1>, vector<256xi8>
