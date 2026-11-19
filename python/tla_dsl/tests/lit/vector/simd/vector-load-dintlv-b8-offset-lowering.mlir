// RUN: %tla_compile %s --mlir-print-ir-after=tla-vector-region-simd -o %t 2>&1 | %filecheck %s --check-prefix=SIMD
// RUN: %filecheck %s --check-prefix=HELPER < %t

!source = !tla.tensor<!tla.layout<!tla.shape<544>, !tla.stride<1>, !tla.shape<544>, RowMajor>, !tla.coord<32>, !tla.ptr<i8, ub, 1>>
!dest = !tla.tensor<!tla.layout<!tla.shape<256>, !tla.stride<1>, !tla.shape<256>, RowMajor>, !tla.coord<0>, !tla.ptr<i8, ub, 1>>

!source_f8 = !tla.tensor<!tla.layout<!tla.shape<544>, !tla.stride<1>, !tla.shape<544>, RowMajor>, !tla.coord<32>, !tla.ptr<f8E4M3FN, ub, 1>>
!dest_f8 = !tla.tensor<!tla.layout<!tla.shape<256>, !tla.stride<1>, !tla.shape<256>, RowMajor>, !tla.coord<0>, !tla.ptr<f8E4M3FN, ub, 1>>

!source_hif8 = !tla.tensor<!tla.layout<!tla.shape<544>, !tla.stride<1>, !tla.shape<544>, RowMajor>, !tla.coord<32>, !tla.ptr<!tla.hif8, ub, 1>>
!dest_hif8 = !tla.tensor<!tla.layout<!tla.shape<256>, !tla.stride<1>, !tla.shape<256>, RowMajor>, !tla.coord<0>, !tla.ptr<!tla.hif8, ub, 1>>

!source_f4e2m1x2 = !tla.tensor<!tla.layout<!tla.shape<544>, !tla.stride<1>, !tla.shape<544>, RowMajor>, !tla.coord<32>, !tla.ptr<!tla.f4e2m1x2, ub, 1>>
!dest_f4e2m1x2 = !tla.tensor<!tla.layout<!tla.shape<256>, !tla.stride<1>, !tla.shape<256>, RowMajor>, !tla.coord<0>, !tla.ptr<!tla.f4e2m1x2, ub, 1>>
!source_f4e1m2x2 = !tla.tensor<!tla.layout<!tla.shape<544>, !tla.stride<1>, !tla.shape<544>, RowMajor>, !tla.coord<32>, !tla.ptr<!tla.f4e1m2x2, ub, 1>>
!dest_f4e1m2x2 = !tla.tensor<!tla.layout<!tla.shape<256>, !tla.stride<1>, !tla.shape<256>, RowMajor>, !tla.coord<0>, !tla.ptr<!tla.f4e1m2x2, ub, 1>>
!source_f8e8m0 = !tla.tensor<!tla.layout<!tla.shape<544>, !tla.stride<1>, !tla.shape<544>, RowMajor>, !tla.coord<32>, !tla.ptr<!tla.f8e8m0, ub, 1>>
!dest_f8e8m0 = !tla.tensor<!tla.layout<!tla.shape<256>, !tla.stride<1>, !tla.shape<256>, RowMajor>, !tla.coord<0>, !tla.ptr<!tla.f8e8m0, ub, 1>>

module {
  func.func @load_dintlv_b8_offset(
      %s: memref<544xi8, #hivm.address_space<ub>>,
      %d0: memref<256xi8, #hivm.address_space<ub>>,
      %d1: memref<256xi8, #hivm.address_space<ub>>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c32 = arith.constant 32 : index
    %c256 = arith.constant 256 : index
    %c544 = arith.constant 544 : index
    %src = tla.tensor_desc %s shape [%c1, %c544, %c1, %c1] stride [%c544, %c1, %c1, %c1] origin_shape [%c1, %c544] coord [%c0, %c32] : memref<544xi8, #hivm.address_space<ub>> -> !source
    %dst0 = tla.tensor_desc %d0 shape [%c1, %c256, %c1, %c1] stride [%c256, %c1, %c1, %c1] origin_shape [%c1, %c256] coord [%c0, %c0] : memref<256xi8, #hivm.address_space<ub>> -> !dest
    %dst1 = tla.tensor_desc %d1 shape [%c1, %c256, %c1, %c1] stride [%c256, %c1, %c1, %c1] origin_shape [%c1, %c256] coord [%c0, %c0] : memref<256xi8, #hivm.address_space<ub>> -> !dest
    "tla.vec.func"() ({
      %v0, %v1 = tla.load %src {load_dist = #tla.load_dist<dintlv_b8>} : !source -> !tla.vector<256xi8>, !tla.vector<256xi8>
      tla.store %dst0, %v0 : !dest, !tla.vector<256xi8>
      tla.store %dst1, %v1 : !dest, !tla.vector<256xi8>
    }) : () -> ()
    return
  }

  func.func @load_dintlv_b8_f8_offset(
      %s: memref<544xf8E4M3FN, #hivm.address_space<ub>>,
      %d0: memref<256xf8E4M3FN, #hivm.address_space<ub>>,
      %d1: memref<256xf8E4M3FN, #hivm.address_space<ub>>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c32 = arith.constant 32 : index
    %c256 = arith.constant 256 : index
    %c544 = arith.constant 544 : index
    %src = tla.tensor_desc %s shape [%c1, %c544, %c1, %c1] stride [%c544, %c1, %c1, %c1] origin_shape [%c1, %c544] coord [%c0, %c32] : memref<544xf8E4M3FN, #hivm.address_space<ub>> -> !source_f8
    %dst0 = tla.tensor_desc %d0 shape [%c1, %c256, %c1, %c1] stride [%c256, %c1, %c1, %c1] origin_shape [%c1, %c256] coord [%c0, %c0] : memref<256xf8E4M3FN, #hivm.address_space<ub>> -> !dest_f8
    %dst1 = tla.tensor_desc %d1 shape [%c1, %c256, %c1, %c1] stride [%c256, %c1, %c1, %c1] origin_shape [%c1, %c256] coord [%c0, %c0] : memref<256xf8E4M3FN, #hivm.address_space<ub>> -> !dest_f8
    "tla.vec.func"() ({
      %v0, %v1 = tla.load %src {load_dist = #tla.load_dist<dintlv_b8>} : !source_f8 -> !tla.vector<256xf8E4M3FN>, !tla.vector<256xf8E4M3FN>
      tla.store %dst0, %v0 : !dest_f8, !tla.vector<256xf8E4M3FN>
      tla.store %dst1, %v1 : !dest_f8, !tla.vector<256xf8E4M3FN>
    }) : () -> ()
    return
  }

  func.func @load_dintlv_b8_hif8_offset(
      %s: memref<544xi8, #hivm.address_space<ub>>,
      %d0: memref<256xi8, #hivm.address_space<ub>>,
      %d1: memref<256xi8, #hivm.address_space<ub>>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c32 = arith.constant 32 : index
    %c256 = arith.constant 256 : index
    %c544 = arith.constant 544 : index
    %src = tla.tensor_desc %s shape [%c1, %c544, %c1, %c1] stride [%c544, %c1, %c1, %c1] origin_shape [%c1, %c544] coord [%c0, %c32] : memref<544xi8, #hivm.address_space<ub>> -> !source_hif8
    %dst0 = tla.tensor_desc %d0 shape [%c1, %c256, %c1, %c1] stride [%c256, %c1, %c1, %c1] origin_shape [%c1, %c256] coord [%c0, %c0] : memref<256xi8, #hivm.address_space<ub>> -> !dest_hif8
    %dst1 = tla.tensor_desc %d1 shape [%c1, %c256, %c1, %c1] stride [%c256, %c1, %c1, %c1] origin_shape [%c1, %c256] coord [%c0, %c0] : memref<256xi8, #hivm.address_space<ub>> -> !dest_hif8
    "tla.vec.func"() ({
      %v0, %v1 = tla.load %src {load_dist = #tla.load_dist<dintlv_b8>} : !source_hif8 -> !tla.vector<256x!tla.hif8>, !tla.vector<256x!tla.hif8>
      tla.store %dst0, %v0 : !dest_hif8, !tla.vector<256x!tla.hif8>
      tla.store %dst1, %v1 : !dest_hif8, !tla.vector<256x!tla.hif8>
    }) : () -> ()
    return
  }

  func.func @load_dintlv_b8_f4e2m1x2_offset(
      %s: memref<544xi8, #hivm.address_space<ub>>,
      %d0: memref<256xi8, #hivm.address_space<ub>>,
      %d1: memref<256xi8, #hivm.address_space<ub>>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c32 = arith.constant 32 : index
    %c256 = arith.constant 256 : index
    %c544 = arith.constant 544 : index
    %src = tla.tensor_desc %s shape [%c1, %c544, %c1, %c1] stride [%c544, %c1, %c1, %c1] origin_shape [%c1, %c544] coord [%c0, %c32] : memref<544xi8, #hivm.address_space<ub>> -> !source_f4e2m1x2
    %dst0 = tla.tensor_desc %d0 shape [%c1, %c256, %c1, %c1] stride [%c256, %c1, %c1, %c1] origin_shape [%c1, %c256] coord [%c0, %c0] : memref<256xi8, #hivm.address_space<ub>> -> !dest_f4e2m1x2
    %dst1 = tla.tensor_desc %d1 shape [%c1, %c256, %c1, %c1] stride [%c256, %c1, %c1, %c1] origin_shape [%c1, %c256] coord [%c0, %c0] : memref<256xi8, #hivm.address_space<ub>> -> !dest_f4e2m1x2
    "tla.vec.func"() ({
      %v0, %v1 = tla.load %src {load_dist = #tla.load_dist<dintlv_b8>} : !source_f4e2m1x2 -> !tla.vector<256x!tla.f4e2m1x2>, !tla.vector<256x!tla.f4e2m1x2>
      tla.store %dst0, %v0 : !dest_f4e2m1x2, !tla.vector<256x!tla.f4e2m1x2>
      tla.store %dst1, %v1 : !dest_f4e2m1x2, !tla.vector<256x!tla.f4e2m1x2>
    }) : () -> ()
    return
  }

  func.func @load_dintlv_b8_f4e1m2x2_offset(
      %s: memref<544xi8, #hivm.address_space<ub>>,
      %d0: memref<256xi8, #hivm.address_space<ub>>,
      %d1: memref<256xi8, #hivm.address_space<ub>>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c32 = arith.constant 32 : index
    %c256 = arith.constant 256 : index
    %c544 = arith.constant 544 : index
    %src = tla.tensor_desc %s shape [%c1, %c544, %c1, %c1] stride [%c544, %c1, %c1, %c1] origin_shape [%c1, %c544] coord [%c0, %c32] : memref<544xi8, #hivm.address_space<ub>> -> !source_f4e1m2x2
    %dst0 = tla.tensor_desc %d0 shape [%c1, %c256, %c1, %c1] stride [%c256, %c1, %c1, %c1] origin_shape [%c1, %c256] coord [%c0, %c0] : memref<256xi8, #hivm.address_space<ub>> -> !dest_f4e1m2x2
    %dst1 = tla.tensor_desc %d1 shape [%c1, %c256, %c1, %c1] stride [%c256, %c1, %c1, %c1] origin_shape [%c1, %c256] coord [%c0, %c0] : memref<256xi8, #hivm.address_space<ub>> -> !dest_f4e1m2x2
    "tla.vec.func"() ({
      %v0, %v1 = tla.load %src {load_dist = #tla.load_dist<dintlv_b8>} : !source_f4e1m2x2 -> !tla.vector<256x!tla.f4e1m2x2>, !tla.vector<256x!tla.f4e1m2x2>
      tla.store %dst0, %v0 : !dest_f4e1m2x2, !tla.vector<256x!tla.f4e1m2x2>
      tla.store %dst1, %v1 : !dest_f4e1m2x2, !tla.vector<256x!tla.f4e1m2x2>
    }) : () -> ()
    return
  }

  func.func @load_dintlv_b8_f8e8m0_offset(
      %s: memref<544xi8, #hivm.address_space<ub>>,
      %d0: memref<256xi8, #hivm.address_space<ub>>,
      %d1: memref<256xi8, #hivm.address_space<ub>>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c32 = arith.constant 32 : index
    %c256 = arith.constant 256 : index
    %c544 = arith.constant 544 : index
    %src = tla.tensor_desc %s shape [%c1, %c544, %c1, %c1] stride [%c544, %c1, %c1, %c1] origin_shape [%c1, %c544] coord [%c0, %c32] : memref<544xi8, #hivm.address_space<ub>> -> !source_f8e8m0
    %dst0 = tla.tensor_desc %d0 shape [%c1, %c256, %c1, %c1] stride [%c256, %c1, %c1, %c1] origin_shape [%c1, %c256] coord [%c0, %c0] : memref<256xi8, #hivm.address_space<ub>> -> !dest_f8e8m0
    %dst1 = tla.tensor_desc %d1 shape [%c1, %c256, %c1, %c1] stride [%c256, %c1, %c1, %c1] origin_shape [%c1, %c256] coord [%c0, %c0] : memref<256xi8, #hivm.address_space<ub>> -> !dest_f8e8m0
    "tla.vec.func"() ({
      %v0, %v1 = tla.load %src {load_dist = #tla.load_dist<dintlv_b8>} : !source_f8e8m0 -> !tla.vector<256x!tla.f8e8m0>, !tla.vector<256x!tla.f8e8m0>
      tla.store %dst0, %v0 : !dest_f8e8m0, !tla.vector<256x!tla.f8e8m0>
      tla.store %dst1, %v1 : !dest_f8e8m0, !tla.vector<256x!tla.f8e8m0>
    }) : () -> ()
    return
  }
}

// SIMD-LABEL: func.func private @vector_region_
// SIMD: arith.constant 32 : index
// SIMD: %[[SOURCE:.*]] = memref.reinterpret_cast
// SIMD: %[[BASE:[^, ]+]], %[[OFFSET:[^, ]+]], {{.*}} = memref.extract_strided_metadata %[[SOURCE]]
// SIMD: %[[ADDRESS:.*]] = memref.extract_aligned_pointer_as_index %[[BASE]]
// SIMD: %[[EFFECTIVE:.*]] = arith.addi %[[ADDRESS]], %[[OFFSET]] : index
// SIMD: %[[I64:.*]] = arith.index_cast %[[EFFECTIVE]] : index to i64
// SIMD: %[[PTR:.*]] = llvm.inttoptr %[[I64]] : i64 to !llvm.ptr<6>
// SIMD: hivm_regbaseintrins.intr.hivm.vldsx2.v256s8{{.*}}%[[PTR]]
// HELPER: hivm_regbaseintrins.intr.hivm.vldsx2.v256s8

// SIMD-LABEL: func.func private @vector_region_
// SIMD: arith.constant 32 : index
// SIMD: %[[SOURCE:.*]] = memref.reinterpret_cast
// SIMD: %[[BASE:[^, ]+]], %[[OFFSET:[^, ]+]], {{.*}} = memref.extract_strided_metadata %[[SOURCE]]
// SIMD: %[[ADDRESS:.*]] = memref.extract_aligned_pointer_as_index %[[BASE]]
// SIMD: %[[EFFECTIVE:.*]] = arith.addi %[[ADDRESS]], %[[OFFSET]] : index
// SIMD: %[[I64:.*]] = arith.index_cast %[[EFFECTIVE]] : index to i64
// SIMD: %[[PTR:.*]] = llvm.inttoptr %[[I64]] : i64 to !llvm.ptr<6>
// SIMD: call @_mlir_ciface_load_dintlv_b8_fp8_e4m3fn(%[[PTR]],
// SIMD: llvm.load
// SIMD-SAME: !llvm.struct<(vector<256xf8E4M3FN>, vector<256xf8E4M3FN>)>
// HELPER: call @_mlir_ciface_load_dintlv_b8_fp8_e4m3fn(

// SIMD-LABEL: func.func private @vector_region_
// SIMD: arith.constant 32 : index
// SIMD: %[[SOURCE:.*]] = memref.reinterpret_cast
// SIMD: %[[BASE:[^, ]+]], %[[OFFSET:[^, ]+]], {{.*}} = memref.extract_strided_metadata %[[SOURCE]]
// SIMD: %[[ADDRESS:.*]] = memref.extract_aligned_pointer_as_index %[[BASE]]
// SIMD: %[[EFFECTIVE:.*]] = arith.addi %[[ADDRESS]], %[[OFFSET]] : index
// SIMD: %[[I64:.*]] = arith.index_cast %[[EFFECTIVE]] : index to i64
// SIMD: %[[PTR:.*]] = llvm.inttoptr %[[I64]] : i64 to !llvm.ptr<6>
// SIMD: call @_mlir_ciface_load_dintlv_b8_hifloat8(%[[PTR]],
// SIMD: llvm.load
// SIMD-SAME: !llvm.struct<(vector<256xi8>, vector<256xi8>)>
// HELPER: call @_mlir_ciface_load_dintlv_b8_hifloat8(

// SIMD-LABEL: func.func private @vector_region_
// SIMD: arith.constant 32 : index
// SIMD: %[[SOURCE:.*]] = memref.reinterpret_cast
// SIMD: %[[BASE:[^, ]+]], %[[OFFSET:[^, ]+]], {{.*}} = memref.extract_strided_metadata %[[SOURCE]]
// SIMD: %[[ADDRESS:.*]] = memref.extract_aligned_pointer_as_index %[[BASE]]
// SIMD: %[[EFFECTIVE:.*]] = arith.addi %[[ADDRESS]], %[[OFFSET]] : index
// SIMD: %[[I64:.*]] = arith.index_cast %[[EFFECTIVE]] : index to i64
// SIMD: %[[PTR:.*]] = llvm.inttoptr %[[I64]] : i64 to !llvm.ptr<6>
// SIMD: call @_mlir_ciface_load_dintlv_b8_fp4x2_e2m1(%[[PTR]],
// SIMD: llvm.load
// SIMD-SAME: !llvm.struct<(vector<256xi8>, vector<256xi8>)>
// HELPER: call @_mlir_ciface_load_dintlv_b8_fp4x2_e2m1(

// SIMD-LABEL: func.func private @vector_region_
// SIMD: arith.constant 32 : index
// SIMD: %[[SOURCE:.*]] = memref.reinterpret_cast
// SIMD: %[[BASE:[^, ]+]], %[[OFFSET:[^, ]+]], {{.*}} = memref.extract_strided_metadata %[[SOURCE]]
// SIMD: %[[ADDRESS:.*]] = memref.extract_aligned_pointer_as_index %[[BASE]]
// SIMD: %[[EFFECTIVE:.*]] = arith.addi %[[ADDRESS]], %[[OFFSET]] : index
// SIMD: %[[I64:.*]] = arith.index_cast %[[EFFECTIVE]] : index to i64
// SIMD: %[[PTR:.*]] = llvm.inttoptr %[[I64]] : i64 to !llvm.ptr<6>
// SIMD: call @_mlir_ciface_load_dintlv_b8_fp4x2_e1m2(%[[PTR]],
// SIMD: llvm.load
// SIMD-SAME: !llvm.struct<(vector<256xi8>, vector<256xi8>)>
// HELPER: call @_mlir_ciface_load_dintlv_b8_fp4x2_e1m2(

// SIMD-LABEL: func.func private @vector_region_
// SIMD: arith.constant 32 : index
// SIMD: %[[SOURCE:.*]] = memref.reinterpret_cast
// SIMD: %[[BASE:[^, ]+]], %[[OFFSET:[^, ]+]], {{.*}} = memref.extract_strided_metadata %[[SOURCE]]
// SIMD: %[[ADDRESS:.*]] = memref.extract_aligned_pointer_as_index %[[BASE]]
// SIMD: %[[EFFECTIVE:.*]] = arith.addi %[[ADDRESS]], %[[OFFSET]] : index
// SIMD: %[[I64:.*]] = arith.index_cast %[[EFFECTIVE]] : index to i64
// SIMD: %[[PTR:.*]] = llvm.inttoptr %[[I64]] : i64 to !llvm.ptr<6>
// SIMD: call @_mlir_ciface_load_dintlv_b8_fp8_e8m0(%[[PTR]],
// SIMD: llvm.load
// SIMD-SAME: !llvm.struct<(vector<256xi8>, vector<256xi8>)>
// HELPER: call @_mlir_ciface_load_dintlv_b8_fp8_e8m0(
