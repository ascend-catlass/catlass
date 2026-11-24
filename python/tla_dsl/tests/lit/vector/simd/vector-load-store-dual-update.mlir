// RUN: %tla_compile %s --mlir-print-ir-after=tla-vector-region-simd -o %t 2>&1 | %filecheck %s

!tile = !tla.tensor<!tla.layout<!tla.shape<256>, !tla.stride<1>, !tla.shape<256>, RowMajor>, !tla.coord<0>, !tla.ptr<bf16, ub, 32>>
!ub = memref<256xbf16, #hivm.address_space<ub>>
!reg = !tla.vector<128xbf16>
!ptr = !tla.ptr<bf16, ub, 32>

module {
  func.func @dual_update(%input: !ub, %output: !ub, %step: i32) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c256 = arith.constant 256 : index
    %src = tla.tensor_desc %input shape [%c1, %c256, %c1, %c1] stride [%c256, %c1, %c1, %c1] origin_shape [%c1, %c256] coord [%c0, %c0] : !ub -> !tile
    %dst = tla.tensor_desc %output shape [%c1, %c256, %c1, %c1] stride [%c256, %c1, %c1, %c1] origin_shape [%c1, %c256] coord [%c0, %c0] : !ub -> !tile
    "tla.vec.func"() ({
      %even, %odd, %next_src = tla.load_update %src post_update_stride %step {load_dist = #tla.load_dist<dintlv_b16>} : !tile post_update_stride i32 -> !reg second !reg, !ptr
      %next_dst = tla.store_update %dst, %even, %odd post_update_stride %step {store_dist = #tla.store_dist<intlv_b16>} : !tile, !reg, !reg post_update_stride i32 -> !ptr
    }) : () -> ()
    return
  }
}

// CHECK-LABEL: func.func private @vector_region_
// CHECK: %[[LOAD_OFF:.*]] = arith.muli {{.*}} : i32
// CHECK: %[[DIST:.*]] = arith.constant 12 : i32
// CHECK: hivm_regbaseintrins.intr.hivm.vldsx2.post.v128u16{{.*}}%[[LOAD_OFF]], %[[DIST]]
// CHECK-SAME: !llvm.struct<(vector<128xi16>, vector<128xi16>, ptr<6>)>
// CHECK: llvm.extractvalue {{.*}}[0]
// CHECK: llvm.extractvalue {{.*}}[1]
// CHECK: llvm.extractvalue {{.*}}[2]
// CHECK: hivm_regbaseintrins.intr.hivm.vstsx2.v128b16
