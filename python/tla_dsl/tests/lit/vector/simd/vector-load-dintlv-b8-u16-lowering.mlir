// RUN: %tla_compile %s --mlir-print-ir-after=tla-vector-region-simd -o %t 2>&1 | %filecheck %s

!source = !tla.tensor<!tla.layout<!tla.shape<272>, !tla.stride<1>, !tla.shape<272>, RowMajor>, !tla.coord<16>, !tla.ptr<ui16, ub, 32>>
!dest = !tla.tensor<!tla.layout<!tla.shape<128>, !tla.stride<1>, !tla.shape<128>, RowMajor>, !tla.coord<0>, !tla.ptr<ui16, ub, 32>>

module {
  func.func @load_dintlv_b8_u16_offset(
      %src_buf: memref<272xui16, #hivm.address_space<ub>>,
      %low_buf: memref<128xui16, #hivm.address_space<ub>>,
      %high_buf: memref<128xui16, #hivm.address_space<ub>>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c16 = arith.constant 16 : index
    %c128 = arith.constant 128 : index
    %c272 = arith.constant 272 : index
    %src = tla.tensor_desc %src_buf shape [%c1, %c272, %c1, %c1] stride [%c272, %c1, %c1, %c1] origin_shape [%c1, %c272] coord [%c0, %c16] : memref<272xui16, #hivm.address_space<ub>> -> !source
    %low_dst = tla.tensor_desc %low_buf shape [%c1, %c128, %c1, %c1] stride [%c128, %c1, %c1, %c1] origin_shape [%c1, %c128] coord [%c0, %c0] : memref<128xui16, #hivm.address_space<ub>> -> !dest
    %high_dst = tla.tensor_desc %high_buf shape [%c1, %c128, %c1, %c1] stride [%c128, %c1, %c1, %c1] origin_shape [%c1, %c128] coord [%c0, %c0] : memref<128xui16, #hivm.address_space<ub>> -> !dest
    "tla.vec.func"() ({
      %low, %high = tla.load %src {load_dist = #tla.load_dist<dintlv_b8>} : !source -> !tla.vector<128xui16>, !tla.vector<128xui16>
      tla.store %low_dst, %low : !dest, !tla.vector<128xui16>
      tla.store %high_dst, %high : !dest, !tla.vector<128xui16>
    }) : () -> ()
    return
  }
}

// CHECK-LABEL: func.func private @vector_region_
// CHECK: arith.constant 16 : index
// CHECK: %[[SOURCE:.*]] = memref.reinterpret_cast
// CHECK: %[[BASE:[^, ]+]], %[[OFFSET:[^, ]+]], {{.*}} = memref.extract_strided_metadata %[[SOURCE]]
// CHECK: %[[ADDRESS:.*]] = memref.extract_aligned_pointer_as_index %[[BASE]]
// CHECK: %[[TWO:.*]] = arith.constant 2 : index
// CHECK: %[[BYTE_OFFSET:.*]] = arith.muli %[[OFFSET]], %[[TWO]] : index
// CHECK: %[[EFFECTIVE:.*]] = arith.addi %[[ADDRESS]], %[[BYTE_OFFSET]] : index
// CHECK: %[[I64:.*]] = arith.index_cast %[[EFFECTIVE]] : index to i64
// CHECK: %[[PTR:.*]] = llvm.inttoptr %[[I64]] : i64 to !llvm.ptr<6>
// CHECK: %[[ZERO:.*]] = arith.constant 0 : i32
// CHECK: %[[DIST:.*]] = arith.constant 11 : i32
// CHECK: %[[PAIR:.*]] = "hivm_regbaseintrins.intr.hivm.vldsx2.v128u16"(%[[PTR]], %[[ZERO]], %[[DIST]], %[[ZERO]])
// CHECK-SAME: !llvm.struct<(vector<128xi16>, vector<128xi16>)>
// CHECK: llvm.extractvalue %[[PAIR]][0]
// CHECK: llvm.extractvalue %[[PAIR]][1]
