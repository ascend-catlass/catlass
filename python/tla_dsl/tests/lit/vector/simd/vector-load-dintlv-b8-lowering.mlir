// RUN: %tla_compile %s -o %t
// RUN: %filecheck %s --implicit-check-not="ave.hir.vload <DINTLV_B8>" --implicit-check-not="call @_mlir_ciface_load_dintlv_b8_int8" --implicit-check-not="call @_mlir_ciface_load_dintlv_b8_uint8" < %t
// RUN: %tla_compile %s --mlir-print-ir-after=tla-vector-region-simd -o %t 2>&1 | %filecheck %s --check-prefix=SIMD --implicit-check-not="vector.bitcast"

!s_i8 = !tla.tensor<!tla.layout<!tla.shape<512>, !tla.stride<1>, !tla.shape<512>, RowMajor>, !tla.coord<0>, !tla.ptr<i8, ub, 1>>
!d_i8 = !tla.tensor<!tla.layout<!tla.shape<256>, !tla.stride<1>, !tla.shape<256>, RowMajor>, !tla.coord<0>, !tla.ptr<i8, ub, 1>>
!s_ui8 = !tla.tensor<!tla.layout<!tla.shape<512>, !tla.stride<1>, !tla.shape<512>, RowMajor>, !tla.coord<0>, !tla.ptr<ui8, ub, 1>>
!d_ui8 = !tla.tensor<!tla.layout<!tla.shape<256>, !tla.stride<1>, !tla.shape<256>, RowMajor>, !tla.coord<0>, !tla.ptr<ui8, ub, 1>>
!s_f8e4m3fn = !tla.tensor<!tla.layout<!tla.shape<512>, !tla.stride<1>, !tla.shape<512>, RowMajor>, !tla.coord<0>, !tla.ptr<f8E4M3FN, ub, 1>>
!d_f8e4m3fn = !tla.tensor<!tla.layout<!tla.shape<256>, !tla.stride<1>, !tla.shape<256>, RowMajor>, !tla.coord<0>, !tla.ptr<f8E4M3FN, ub, 1>>
!s_f8e5m2 = !tla.tensor<!tla.layout<!tla.shape<512>, !tla.stride<1>, !tla.shape<512>, RowMajor>, !tla.coord<0>, !tla.ptr<f8E5M2, ub, 1>>
!d_f8e5m2 = !tla.tensor<!tla.layout<!tla.shape<256>, !tla.stride<1>, !tla.shape<256>, RowMajor>, !tla.coord<0>, !tla.ptr<f8E5M2, ub, 1>>

!s_hif8 = !tla.tensor<!tla.layout<!tla.shape<512>, !tla.stride<1>, !tla.shape<512>, RowMajor>, !tla.coord<0>, !tla.ptr<!tla.hif8, ub, 1>>
!d_hif8 = !tla.tensor<!tla.layout<!tla.shape<256>, !tla.stride<1>, !tla.shape<256>, RowMajor>, !tla.coord<0>, !tla.ptr<!tla.hif8, ub, 1>>

!s_f4e2m1x2 = !tla.tensor<!tla.layout<!tla.shape<512>, !tla.stride<1>, !tla.shape<512>, RowMajor>, !tla.coord<0>, !tla.ptr<!tla.f4e2m1x2, ub, 1>>
!d_f4e2m1x2 = !tla.tensor<!tla.layout<!tla.shape<256>, !tla.stride<1>, !tla.shape<256>, RowMajor>, !tla.coord<0>, !tla.ptr<!tla.f4e2m1x2, ub, 1>>
!s_f4e1m2x2 = !tla.tensor<!tla.layout<!tla.shape<512>, !tla.stride<1>, !tla.shape<512>, RowMajor>, !tla.coord<0>, !tla.ptr<!tla.f4e1m2x2, ub, 1>>
!d_f4e1m2x2 = !tla.tensor<!tla.layout<!tla.shape<256>, !tla.stride<1>, !tla.shape<256>, RowMajor>, !tla.coord<0>, !tla.ptr<!tla.f4e1m2x2, ub, 1>>
!s_f8e8m0 = !tla.tensor<!tla.layout<!tla.shape<512>, !tla.stride<1>, !tla.shape<512>, RowMajor>, !tla.coord<0>, !tla.ptr<!tla.f8e8m0, ub, 1>>
!d_f8e8m0 = !tla.tensor<!tla.layout<!tla.shape<256>, !tla.stride<1>, !tla.shape<256>, RowMajor>, !tla.coord<0>, !tla.ptr<!tla.f8e8m0, ub, 1>>

module {
  func.func @load_dintlv_b8_i8(%s: memref<512xi8, #hivm.address_space<ub>>, %d0: memref<256xi8, #hivm.address_space<ub>>, %d1: memref<256xi8, #hivm.address_space<ub>>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c512 = arith.constant 512 : index
    %c256 = arith.constant 256 : index
    %src = tla.tensor_desc %s shape [%c1, %c512, %c1, %c1] stride [%c512, %c1, %c1, %c1] origin_shape [%c1, %c512] coord [%c0, %c0] : memref<512xi8, #hivm.address_space<ub>> -> !s_i8
    %dst0 = tla.tensor_desc %d0 shape [%c1, %c256, %c1, %c1] stride [%c256, %c1, %c1, %c1] origin_shape [%c1, %c256] coord [%c0, %c0] : memref<256xi8, #hivm.address_space<ub>> -> !d_i8
    %dst1 = tla.tensor_desc %d1 shape [%c1, %c256, %c1, %c1] stride [%c256, %c1, %c1, %c1] origin_shape [%c1, %c256] coord [%c0, %c0] : memref<256xi8, #hivm.address_space<ub>> -> !d_i8
    "tla.vec.func"() ({
      %v0, %v1 = tla.load %src {load_dist = #tla.load_dist<dintlv_b8>} : !s_i8 -> !tla.vector<256xi8>, !tla.vector<256xi8>
      tla.store %dst0, %v0 : !d_i8, !tla.vector<256xi8>
      tla.store %dst1, %v1 : !d_i8, !tla.vector<256xi8>
    }) : () -> ()
    return
  }

  func.func @load_dintlv_b8_ui8(%s: memref<512xui8, #hivm.address_space<ub>>, %d0: memref<256xui8, #hivm.address_space<ub>>, %d1: memref<256xui8, #hivm.address_space<ub>>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c512 = arith.constant 512 : index
    %c256 = arith.constant 256 : index
    %src = tla.tensor_desc %s shape [%c1, %c512, %c1, %c1] stride [%c512, %c1, %c1, %c1] origin_shape [%c1, %c512] coord [%c0, %c0] : memref<512xui8, #hivm.address_space<ub>> -> !s_ui8
    %dst0 = tla.tensor_desc %d0 shape [%c1, %c256, %c1, %c1] stride [%c256, %c1, %c1, %c1] origin_shape [%c1, %c256] coord [%c0, %c0] : memref<256xui8, #hivm.address_space<ub>> -> !d_ui8
    %dst1 = tla.tensor_desc %d1 shape [%c1, %c256, %c1, %c1] stride [%c256, %c1, %c1, %c1] origin_shape [%c1, %c256] coord [%c0, %c0] : memref<256xui8, #hivm.address_space<ub>> -> !d_ui8
    "tla.vec.func"() ({
      %v0, %v1 = tla.load %src {load_dist = #tla.load_dist<dintlv_b8>} : !s_ui8 -> !tla.vector<256xui8>, !tla.vector<256xui8>
      tla.store %dst0, %v0 : !d_ui8, !tla.vector<256xui8>
      tla.store %dst1, %v1 : !d_ui8, !tla.vector<256xui8>
    }) : () -> ()
    return
  }

  func.func @load_dintlv_b8_f8e4m3fn(%s: memref<512xf8E4M3FN, #hivm.address_space<ub>>, %d0: memref<256xf8E4M3FN, #hivm.address_space<ub>>, %d1: memref<256xf8E4M3FN, #hivm.address_space<ub>>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c512 = arith.constant 512 : index
    %c256 = arith.constant 256 : index
    %src = tla.tensor_desc %s shape [%c1, %c512, %c1, %c1] stride [%c512, %c1, %c1, %c1] origin_shape [%c1, %c512] coord [%c0, %c0] : memref<512xf8E4M3FN, #hivm.address_space<ub>> -> !s_f8e4m3fn
    %dst0 = tla.tensor_desc %d0 shape [%c1, %c256, %c1, %c1] stride [%c256, %c1, %c1, %c1] origin_shape [%c1, %c256] coord [%c0, %c0] : memref<256xf8E4M3FN, #hivm.address_space<ub>> -> !d_f8e4m3fn
    %dst1 = tla.tensor_desc %d1 shape [%c1, %c256, %c1, %c1] stride [%c256, %c1, %c1, %c1] origin_shape [%c1, %c256] coord [%c0, %c0] : memref<256xf8E4M3FN, #hivm.address_space<ub>> -> !d_f8e4m3fn
    "tla.vec.func"() ({
      %v0, %v1 = tla.load %src {load_dist = #tla.load_dist<dintlv_b8>} : !s_f8e4m3fn -> !tla.vector<256xf8E4M3FN>, !tla.vector<256xf8E4M3FN>
      tla.store %dst0, %v0 : !d_f8e4m3fn, !tla.vector<256xf8E4M3FN>
      tla.store %dst1, %v1 : !d_f8e4m3fn, !tla.vector<256xf8E4M3FN>
    }) : () -> ()
    return
  }

  func.func @load_dintlv_b8_f8e5m2(%s: memref<512xf8E5M2, #hivm.address_space<ub>>, %d0: memref<256xf8E5M2, #hivm.address_space<ub>>, %d1: memref<256xf8E5M2, #hivm.address_space<ub>>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c512 = arith.constant 512 : index
    %c256 = arith.constant 256 : index
    %src = tla.tensor_desc %s shape [%c1, %c512, %c1, %c1] stride [%c512, %c1, %c1, %c1] origin_shape [%c1, %c512] coord [%c0, %c0] : memref<512xf8E5M2, #hivm.address_space<ub>> -> !s_f8e5m2
    %dst0 = tla.tensor_desc %d0 shape [%c1, %c256, %c1, %c1] stride [%c256, %c1, %c1, %c1] origin_shape [%c1, %c256] coord [%c0, %c0] : memref<256xf8E5M2, #hivm.address_space<ub>> -> !d_f8e5m2
    %dst1 = tla.tensor_desc %d1 shape [%c1, %c256, %c1, %c1] stride [%c256, %c1, %c1, %c1] origin_shape [%c1, %c256] coord [%c0, %c0] : memref<256xf8E5M2, #hivm.address_space<ub>> -> !d_f8e5m2
    "tla.vec.func"() ({
      %v0, %v1 = tla.load %src {load_dist = #tla.load_dist<dintlv_b8>} : !s_f8e5m2 -> !tla.vector<256xf8E5M2>, !tla.vector<256xf8E5M2>
      tla.store %dst0, %v0 : !d_f8e5m2, !tla.vector<256xf8E5M2>
      tla.store %dst1, %v1 : !d_f8e5m2, !tla.vector<256xf8E5M2>
    }) : () -> ()
    return
  }

  func.func @load_dintlv_b8_hif8(%s: memref<512xi8, #hivm.address_space<ub>>, %d0: memref<256xi8, #hivm.address_space<ub>>, %d1: memref<256xi8, #hivm.address_space<ub>>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c512 = arith.constant 512 : index
    %c256 = arith.constant 256 : index
    %src = tla.tensor_desc %s shape [%c1, %c512, %c1, %c1] stride [%c512, %c1, %c1, %c1] origin_shape [%c1, %c512] coord [%c0, %c0] : memref<512xi8, #hivm.address_space<ub>> -> !s_hif8
    %dst0 = tla.tensor_desc %d0 shape [%c1, %c256, %c1, %c1] stride [%c256, %c1, %c1, %c1] origin_shape [%c1, %c256] coord [%c0, %c0] : memref<256xi8, #hivm.address_space<ub>> -> !d_hif8
    %dst1 = tla.tensor_desc %d1 shape [%c1, %c256, %c1, %c1] stride [%c256, %c1, %c1, %c1] origin_shape [%c1, %c256] coord [%c0, %c0] : memref<256xi8, #hivm.address_space<ub>> -> !d_hif8
    "tla.vec.func"() ({
      %v0, %v1 = tla.load %src {load_dist = #tla.load_dist<dintlv_b8>} : !s_hif8 -> !tla.vector<256x!tla.hif8>, !tla.vector<256x!tla.hif8>
      tla.store %dst0, %v0 : !d_hif8, !tla.vector<256x!tla.hif8>
      tla.store %dst1, %v1 : !d_hif8, !tla.vector<256x!tla.hif8>
    }) : () -> ()
    return
  }

  func.func @load_dintlv_b8_f4e2m1x2(%s: memref<512xi8, #hivm.address_space<ub>>, %d0: memref<256xi8, #hivm.address_space<ub>>, %d1: memref<256xi8, #hivm.address_space<ub>>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c512 = arith.constant 512 : index
    %c256 = arith.constant 256 : index
    %src = tla.tensor_desc %s shape [%c1, %c512, %c1, %c1] stride [%c512, %c1, %c1, %c1] origin_shape [%c1, %c512] coord [%c0, %c0] : memref<512xi8, #hivm.address_space<ub>> -> !s_f4e2m1x2
    %dst0 = tla.tensor_desc %d0 shape [%c1, %c256, %c1, %c1] stride [%c256, %c1, %c1, %c1] origin_shape [%c1, %c256] coord [%c0, %c0] : memref<256xi8, #hivm.address_space<ub>> -> !d_f4e2m1x2
    %dst1 = tla.tensor_desc %d1 shape [%c1, %c256, %c1, %c1] stride [%c256, %c1, %c1, %c1] origin_shape [%c1, %c256] coord [%c0, %c0] : memref<256xi8, #hivm.address_space<ub>> -> !d_f4e2m1x2
    "tla.vec.func"() ({
      %v0, %v1 = tla.load %src {load_dist = #tla.load_dist<dintlv_b8>} : !s_f4e2m1x2 -> !tla.vector<256x!tla.f4e2m1x2>, !tla.vector<256x!tla.f4e2m1x2>
      tla.store %dst0, %v0 : !d_f4e2m1x2, !tla.vector<256x!tla.f4e2m1x2>
      tla.store %dst1, %v1 : !d_f4e2m1x2, !tla.vector<256x!tla.f4e2m1x2>
    }) : () -> ()
    return
  }

  func.func @load_dintlv_b8_f4e1m2x2(%s: memref<512xi8, #hivm.address_space<ub>>, %d0: memref<256xi8, #hivm.address_space<ub>>, %d1: memref<256xi8, #hivm.address_space<ub>>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c512 = arith.constant 512 : index
    %c256 = arith.constant 256 : index
    %src = tla.tensor_desc %s shape [%c1, %c512, %c1, %c1] stride [%c512, %c1, %c1, %c1] origin_shape [%c1, %c512] coord [%c0, %c0] : memref<512xi8, #hivm.address_space<ub>> -> !s_f4e1m2x2
    %dst0 = tla.tensor_desc %d0 shape [%c1, %c256, %c1, %c1] stride [%c256, %c1, %c1, %c1] origin_shape [%c1, %c256] coord [%c0, %c0] : memref<256xi8, #hivm.address_space<ub>> -> !d_f4e1m2x2
    %dst1 = tla.tensor_desc %d1 shape [%c1, %c256, %c1, %c1] stride [%c256, %c1, %c1, %c1] origin_shape [%c1, %c256] coord [%c0, %c0] : memref<256xi8, #hivm.address_space<ub>> -> !d_f4e1m2x2
    "tla.vec.func"() ({
      %v0, %v1 = tla.load %src {load_dist = #tla.load_dist<dintlv_b8>} : !s_f4e1m2x2 -> !tla.vector<256x!tla.f4e1m2x2>, !tla.vector<256x!tla.f4e1m2x2>
      tla.store %dst0, %v0 : !d_f4e1m2x2, !tla.vector<256x!tla.f4e1m2x2>
      tla.store %dst1, %v1 : !d_f4e1m2x2, !tla.vector<256x!tla.f4e1m2x2>
    }) : () -> ()
    return
  }

  func.func @load_dintlv_b8_f8e8m0(%s: memref<512xi8, #hivm.address_space<ub>>, %d0: memref<256xi8, #hivm.address_space<ub>>, %d1: memref<256xi8, #hivm.address_space<ub>>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c512 = arith.constant 512 : index
    %c256 = arith.constant 256 : index
    %src = tla.tensor_desc %s shape [%c1, %c512, %c1, %c1] stride [%c512, %c1, %c1, %c1] origin_shape [%c1, %c512] coord [%c0, %c0] : memref<512xi8, #hivm.address_space<ub>> -> !s_f8e8m0
    %dst0 = tla.tensor_desc %d0 shape [%c1, %c256, %c1, %c1] stride [%c256, %c1, %c1, %c1] origin_shape [%c1, %c256] coord [%c0, %c0] : memref<256xi8, #hivm.address_space<ub>> -> !d_f8e8m0
    %dst1 = tla.tensor_desc %d1 shape [%c1, %c256, %c1, %c1] stride [%c256, %c1, %c1, %c1] origin_shape [%c1, %c256] coord [%c0, %c0] : memref<256xi8, #hivm.address_space<ub>> -> !d_f8e8m0
    "tla.vec.func"() ({
      %v0, %v1 = tla.load %src {load_dist = #tla.load_dist<dintlv_b8>} : !s_f8e8m0 -> !tla.vector<256x!tla.f8e8m0>, !tla.vector<256x!tla.f8e8m0>
      tla.store %dst0, %v0 : !d_f8e8m0, !tla.vector<256x!tla.f8e8m0>
      tla.store %dst1, %v1 : !d_f8e8m0, !tla.vector<256x!tla.f8e8m0>
    }) : () -> ()
    return
  }

}

// CHECK-LABEL: func.func private @vector_region_
// CHECK-NOT: llvm.alloca
// CHECK: %[[PAIR:.*]] = "hivm_regbaseintrins.intr.hivm.vldsx2.v256s8"
// CHECK-SAME: !llvm.struct<(vector<256xi8>, vector<256xi8>)>
// CHECK: llvm.extractvalue %[[PAIR]][0]
// CHECK: llvm.extractvalue %[[PAIR]][1]

// SIMD-LABEL: func.func private @vector_region_
// SIMD-NOT: llvm.alloca
// SIMD: %[[PAIR:.*]] = "hivm_regbaseintrins.intr.hivm.vldsx2.v256s8"
// SIMD-SAME: !llvm.struct<(vector<256xi8>, vector<256xi8>)>
// SIMD: llvm.extractvalue %[[PAIR]][0]
// SIMD: llvm.extractvalue %[[PAIR]][1]

// CHECK-LABEL: func.func private @vector_region_
// CHECK-NOT: llvm.alloca
// CHECK: %[[PAIR:.*]] = "hivm_regbaseintrins.intr.hivm.vldsx2.v256u8"
// CHECK-SAME: !llvm.struct<(vector<256xi8>, vector<256xi8>)>
// CHECK: llvm.extractvalue %[[PAIR]][0]
// CHECK: llvm.extractvalue %[[PAIR]][1]

// SIMD-LABEL: func.func private @vector_region_
// SIMD-NOT: llvm.alloca
// SIMD: %[[PAIR:.*]] = "hivm_regbaseintrins.intr.hivm.vldsx2.v256u8"
// SIMD-SAME: !llvm.struct<(vector<256xi8>, vector<256xi8>)>
// SIMD: llvm.extractvalue %[[PAIR]][0]
// SIMD: llvm.extractvalue %[[PAIR]][1]

// CHECK-LABEL: func.func private @vector_region_
// CHECK: call @_mlir_ciface_load_dintlv_b8_fp8_e4m3fn(
// CHECK: %[[PAIR:.*]] = llvm.load
// CHECK-SAME: !llvm.struct<(vector<256xf8E4M3FN>, vector<256xf8E4M3FN>)>
// CHECK: llvm.extractvalue %[[PAIR]][0]
// CHECK: llvm.extractvalue %[[PAIR]][1]
// SIMD-LABEL: func.func private @vector_region_
// SIMD: llvm.alloca
// SIMD-SAME: alignment = 256
// SIMD: call @_mlir_ciface_load_dintlv_b8_fp8_e4m3fn(
// SIMD: llvm.load
// SIMD-SAME: !llvm.struct<(vector<256xf8E4M3FN>, vector<256xf8E4M3FN>)>

// CHECK-LABEL: func.func private @vector_region_
// CHECK: call @_mlir_ciface_load_dintlv_b8_fp8_e5m2(
// CHECK: %[[PAIR:.*]] = llvm.load
// CHECK-SAME: !llvm.struct<(vector<256xf8E5M2>, vector<256xf8E5M2>)>
// CHECK: llvm.extractvalue %[[PAIR]][0]
// CHECK: llvm.extractvalue %[[PAIR]][1]
// SIMD-LABEL: func.func private @vector_region_
// SIMD: llvm.alloca
// SIMD-SAME: alignment = 256
// SIMD: call @_mlir_ciface_load_dintlv_b8_fp8_e5m2(
// SIMD: llvm.load
// SIMD-SAME: !llvm.struct<(vector<256xf8E5M2>, vector<256xf8E5M2>)>

// CHECK-LABEL: func.func private @vector_region_
// CHECK: call @_mlir_ciface_load_dintlv_b8_hifloat8(
// CHECK: %[[PAIR:.*]] = llvm.load
// CHECK-SAME: !llvm.struct<(vector<256xi8>, vector<256xi8>)>
// CHECK: llvm.extractvalue %[[PAIR]][0]
// CHECK: llvm.extractvalue %[[PAIR]][1]
// SIMD-LABEL: func.func private @vector_region_
// SIMD: llvm.alloca
// SIMD-SAME: alignment = 256
// SIMD: call @_mlir_ciface_load_dintlv_b8_hifloat8(
// SIMD: llvm.load
// SIMD-SAME: !llvm.struct<(vector<256xi8>, vector<256xi8>)>

// CHECK-LABEL: func.func private @vector_region_
// CHECK: call @_mlir_ciface_load_dintlv_b8_fp4x2_e2m1(
// CHECK: %[[PAIR:.*]] = llvm.load
// CHECK-SAME: !llvm.struct<(vector<256xi8>, vector<256xi8>)>
// CHECK: llvm.extractvalue %[[PAIR]][0]
// CHECK: llvm.extractvalue %[[PAIR]][1]
// SIMD-LABEL: func.func private @vector_region_
// SIMD: llvm.alloca
// SIMD-SAME: alignment = 256
// SIMD: call @_mlir_ciface_load_dintlv_b8_fp4x2_e2m1(
// SIMD: llvm.load
// SIMD-SAME: !llvm.struct<(vector<256xi8>, vector<256xi8>)>

// CHECK-LABEL: func.func private @vector_region_
// CHECK: call @_mlir_ciface_load_dintlv_b8_fp4x2_e1m2(
// CHECK: %[[PAIR:.*]] = llvm.load
// CHECK-SAME: !llvm.struct<(vector<256xi8>, vector<256xi8>)>
// CHECK: llvm.extractvalue %[[PAIR]][0]
// CHECK: llvm.extractvalue %[[PAIR]][1]
// SIMD-LABEL: func.func private @vector_region_
// SIMD: llvm.alloca
// SIMD-SAME: alignment = 256
// SIMD: call @_mlir_ciface_load_dintlv_b8_fp4x2_e1m2(
// SIMD: llvm.load
// SIMD-SAME: !llvm.struct<(vector<256xi8>, vector<256xi8>)>

// CHECK-LABEL: func.func private @vector_region_
// CHECK: call @_mlir_ciface_load_dintlv_b8_fp8_e8m0(
// CHECK: %[[PAIR:.*]] = llvm.load
// CHECK-SAME: !llvm.struct<(vector<256xi8>, vector<256xi8>)>
// CHECK: llvm.extractvalue %[[PAIR]][0]
// CHECK: llvm.extractvalue %[[PAIR]][1]
// SIMD-LABEL: func.func private @vector_region_
// SIMD: llvm.alloca
// SIMD-SAME: alignment = 256
// SIMD: call @_mlir_ciface_load_dintlv_b8_fp8_e8m0(
// SIMD: llvm.load
// SIMD-SAME: !llvm.struct<(vector<256xi8>, vector<256xi8>)>
