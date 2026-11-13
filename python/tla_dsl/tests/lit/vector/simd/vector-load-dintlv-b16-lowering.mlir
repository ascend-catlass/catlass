// RUN: %tla_compile %s --mlir-print-ir-after=tla-vector-region-simd -o %t 2>&1 | %filecheck %s

!source_bf16 = !tla.tensor<!tla.layout<!tla.shape<256>, !tla.stride<1>, !tla.shape<256>, RowMajor>, !tla.coord<0>, !tla.ptr<bf16, ub, 32>>
!destination_bf16 = !tla.tensor<!tla.layout<!tla.shape<128>, !tla.stride<1>, !tla.shape<128>, RowMajor>, !tla.coord<0>, !tla.ptr<bf16, ub, 32>>

!source_f16 = !tla.tensor<!tla.layout<!tla.shape<256>, !tla.stride<1>, !tla.shape<256>, RowMajor>, !tla.coord<0>, !tla.ptr<f16, ub, 32>>
!destination_f16 = !tla.tensor<!tla.layout<!tla.shape<128>, !tla.stride<1>, !tla.shape<128>, RowMajor>, !tla.coord<0>, !tla.ptr<f16, ub, 32>>

!source_i16 = !tla.tensor<!tla.layout<!tla.shape<256>, !tla.stride<1>, !tla.shape<256>, RowMajor>, !tla.coord<0>, !tla.ptr<i16, ub, 32>>
!destination_i16 = !tla.tensor<!tla.layout<!tla.shape<128>, !tla.stride<1>, !tla.shape<128>, RowMajor>, !tla.coord<0>, !tla.ptr<i16, ub, 32>>

!source_u16 = !tla.tensor<!tla.layout<!tla.shape<256>, !tla.stride<1>, !tla.shape<256>, RowMajor>, !tla.coord<0>, !tla.ptr<ui16, ub, 32>>
!destination_u16 = !tla.tensor<!tla.layout<!tla.shape<128>, !tla.stride<1>, !tla.shape<128>, RowMajor>, !tla.coord<0>, !tla.ptr<ui16, ub, 32>>

module {
  func.func @load_dintlv_bf16(%source: memref<256xbf16, #hivm.address_space<ub>>, %even: memref<128xbf16, #hivm.address_space<ub>>, %odd: memref<128xbf16, #hivm.address_space<ub>>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c128 = arith.constant 128 : index
    %c256 = arith.constant 256 : index
    %src = tla.tensor_desc %source shape [%c1, %c256, %c1, %c1] stride [%c256, %c1, %c1, %c1] origin_shape [%c1, %c256] coord [%c0, %c0] : memref<256xbf16, #hivm.address_space<ub>> -> !source_bf16
    %dst0 = tla.tensor_desc %even shape [%c1, %c128, %c1, %c1] stride [%c128, %c1, %c1, %c1] origin_shape [%c1, %c128] coord [%c0, %c0] : memref<128xbf16, #hivm.address_space<ub>> -> !destination_bf16
    %dst1 = tla.tensor_desc %odd shape [%c1, %c128, %c1, %c1] stride [%c128, %c1, %c1, %c1] origin_shape [%c1, %c128] coord [%c0, %c0] : memref<128xbf16, #hivm.address_space<ub>> -> !destination_bf16
    "tla.vec.func"() ({
      %even_reg, %odd_reg = tla.load %src {load_dist = #tla.load_dist<dintlv_b16>} : !source_bf16 -> !tla.vector<128xbf16>, !tla.vector<128xbf16>
      tla.store %dst0, %even_reg : !destination_bf16, !tla.vector<128xbf16>
      tla.store %dst1, %odd_reg : !destination_bf16, !tla.vector<128xbf16>
    }) : () -> ()
    return
  }
  func.func @load_dintlv_f16(%source: memref<256xf16, #hivm.address_space<ub>>, %even: memref<128xf16, #hivm.address_space<ub>>, %odd: memref<128xf16, #hivm.address_space<ub>>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c128 = arith.constant 128 : index
    %c256 = arith.constant 256 : index
    %src = tla.tensor_desc %source shape [%c1, %c256, %c1, %c1] stride [%c256, %c1, %c1, %c1] origin_shape [%c1, %c256] coord [%c0, %c0] : memref<256xf16, #hivm.address_space<ub>> -> !source_f16
    %dst0 = tla.tensor_desc %even shape [%c1, %c128, %c1, %c1] stride [%c128, %c1, %c1, %c1] origin_shape [%c1, %c128] coord [%c0, %c0] : memref<128xf16, #hivm.address_space<ub>> -> !destination_f16
    %dst1 = tla.tensor_desc %odd shape [%c1, %c128, %c1, %c1] stride [%c128, %c1, %c1, %c1] origin_shape [%c1, %c128] coord [%c0, %c0] : memref<128xf16, #hivm.address_space<ub>> -> !destination_f16
    "tla.vec.func"() ({
      %even_reg, %odd_reg = tla.load %src {load_dist = #tla.load_dist<dintlv_b16>} : !source_f16 -> !tla.vector<128xf16>, !tla.vector<128xf16>
      tla.store %dst0, %even_reg : !destination_f16, !tla.vector<128xf16>
      tla.store %dst1, %odd_reg : !destination_f16, !tla.vector<128xf16>
    }) : () -> ()
    return
  }
  func.func @load_dintlv_i16(%source: memref<256xi16, #hivm.address_space<ub>>, %even: memref<128xi16, #hivm.address_space<ub>>, %odd: memref<128xi16, #hivm.address_space<ub>>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c128 = arith.constant 128 : index
    %c256 = arith.constant 256 : index
    %src = tla.tensor_desc %source shape [%c1, %c256, %c1, %c1] stride [%c256, %c1, %c1, %c1] origin_shape [%c1, %c256] coord [%c0, %c0] : memref<256xi16, #hivm.address_space<ub>> -> !source_i16
    %dst0 = tla.tensor_desc %even shape [%c1, %c128, %c1, %c1] stride [%c128, %c1, %c1, %c1] origin_shape [%c1, %c128] coord [%c0, %c0] : memref<128xi16, #hivm.address_space<ub>> -> !destination_i16
    %dst1 = tla.tensor_desc %odd shape [%c1, %c128, %c1, %c1] stride [%c128, %c1, %c1, %c1] origin_shape [%c1, %c128] coord [%c0, %c0] : memref<128xi16, #hivm.address_space<ub>> -> !destination_i16
    "tla.vec.func"() ({
      %even_reg, %odd_reg = tla.load %src {load_dist = #tla.load_dist<dintlv_b16>} : !source_i16 -> !tla.vector<128xi16>, !tla.vector<128xi16>
      tla.store %dst0, %even_reg : !destination_i16, !tla.vector<128xi16>
      tla.store %dst1, %odd_reg : !destination_i16, !tla.vector<128xi16>
    }) : () -> ()
    return
  }
  func.func @load_dintlv_u16(%source: memref<256xui16, #hivm.address_space<ub>>, %even: memref<128xui16, #hivm.address_space<ub>>, %odd: memref<128xui16, #hivm.address_space<ub>>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c128 = arith.constant 128 : index
    %c256 = arith.constant 256 : index
    %src = tla.tensor_desc %source shape [%c1, %c256, %c1, %c1] stride [%c256, %c1, %c1, %c1] origin_shape [%c1, %c256] coord [%c0, %c0] : memref<256xui16, #hivm.address_space<ub>> -> !source_u16
    %dst0 = tla.tensor_desc %even shape [%c1, %c128, %c1, %c1] stride [%c128, %c1, %c1, %c1] origin_shape [%c1, %c128] coord [%c0, %c0] : memref<128xui16, #hivm.address_space<ub>> -> !destination_u16
    %dst1 = tla.tensor_desc %odd shape [%c1, %c128, %c1, %c1] stride [%c128, %c1, %c1, %c1] origin_shape [%c1, %c128] coord [%c0, %c0] : memref<128xui16, #hivm.address_space<ub>> -> !destination_u16
    "tla.vec.func"() ({
      %even_reg, %odd_reg = tla.load %src {load_dist = #tla.load_dist<dintlv_b16>} : !source_u16 -> !tla.vector<128xui16>, !tla.vector<128xui16>
      tla.store %dst0, %even_reg : !destination_u16, !tla.vector<128xui16>
      tla.store %dst1, %odd_reg : !destination_u16, !tla.vector<128xui16>
    }) : () -> ()
    return
  }
}

// CHECK-LABEL: func.func private @vector_region_
// CHECK: memref.extract_strided_metadata
// CHECK: %[[ELEMENT_BYTES_BF16:.+]] = arith.constant 2 : index
// CHECK: %[[OFFSET_BF16:.+]] = arith.muli {{.*}}, %[[ELEMENT_BYTES_BF16]]
// CHECK: arith.addi {{.*}}, %[[OFFSET_BF16]]
// CHECK: %[[DIST_BF16:.+]] = arith.constant 12 : i32
// CHECK: hivm_regbaseintrins.intr.hivm.vldsx2.v128bf16{{.*}}%[[DIST_BF16]]
// CHECK-SAME: !llvm.struct<(vector<128xbf16>, vector<128xbf16>)>
// CHECK: llvm.extractvalue {{.*}}[0]
// CHECK: llvm.extractvalue {{.*}}[1]

// CHECK-LABEL: func.func private @vector_region_
// CHECK: memref.extract_strided_metadata
// CHECK: %[[ELEMENT_BYTES_F16:.+]] = arith.constant 2 : index
// CHECK: %[[OFFSET_F16:.+]] = arith.muli {{.*}}, %[[ELEMENT_BYTES_F16]]
// CHECK: arith.addi {{.*}}, %[[OFFSET_F16]]
// CHECK: %[[DIST_F16:.+]] = arith.constant 12 : i32
// CHECK: hivm_regbaseintrins.intr.hivm.vldsx2.v128f16{{.*}}%[[DIST_F16]]
// CHECK-SAME: !llvm.struct<(vector<128xf16>, vector<128xf16>)>
// CHECK: llvm.extractvalue {{.*}}[0]
// CHECK: llvm.extractvalue {{.*}}[1]

// CHECK-LABEL: func.func private @vector_region_
// CHECK: memref.extract_strided_metadata
// CHECK: %[[ELEMENT_BYTES_I16:.+]] = arith.constant 2 : index
// CHECK: %[[OFFSET_I16:.+]] = arith.muli {{.*}}, %[[ELEMENT_BYTES_I16]]
// CHECK: arith.addi {{.*}}, %[[OFFSET_I16]]
// CHECK: %[[DIST_I16:.+]] = arith.constant 12 : i32
// CHECK: hivm_regbaseintrins.intr.hivm.vldsx2.v128u16{{.*}}%[[DIST_I16]]
// CHECK-SAME: !llvm.struct<(vector<128xi16>, vector<128xi16>)>
// CHECK: llvm.extractvalue {{.*}}[0]
// CHECK: llvm.extractvalue {{.*}}[1]

// CHECK-LABEL: func.func private @vector_region_
// CHECK: memref.extract_strided_metadata
// CHECK: %[[ELEMENT_BYTES_U16:.+]] = arith.constant 2 : index
// CHECK: %[[OFFSET_U16:.+]] = arith.muli {{.*}}, %[[ELEMENT_BYTES_U16]]
// CHECK: arith.addi {{.*}}, %[[OFFSET_U16]]
// CHECK: %[[DIST_U16:.+]] = arith.constant 12 : i32
// CHECK: hivm_regbaseintrins.intr.hivm.vldsx2.v128u16{{.*}}%[[DIST_U16]]
// CHECK-SAME: !llvm.struct<(vector<128xi16>, vector<128xi16>)>
// CHECK: llvm.extractvalue {{.*}}[0]
// CHECK: llvm.extractvalue {{.*}}[1]
