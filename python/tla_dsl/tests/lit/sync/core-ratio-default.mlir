// RUN: %tla_compile %s -o %t --mlir-print-ir-after=tla-lower-func 2>&1 | %filecheck %s

!gm = !tla.tensor<!tla.layout<!tla.shape<32,32>, !tla.stride<32,1>, !tla.shape<32,32>, RowMajor>, !tla.coord<0,0>, !tla.ptr<f32, gm, 4>>
!l1 = !tla.tensor<!tla.layout<!tla.shape<(16,2),(8,4)>, !tla.stride<(8,128),(1,256)>, !tla.shape<32,32>, zN>, !tla.coord<0,0>, !tla.ptr<f32, l1, 512>>
!ub = !tla.tensor<!tla.layout<!tla.shape<32,32>, !tla.stride<32,1>, !tla.shape<32,32>, RowMajor>, !tla.coord<0,0>, !tla.ptr<f32, ub, 256>>

module {
  tla.func @core_ratio_default(%gm: !gm) {
    %l1_ptr = tla.alloc_ptr{size_bytes = 4096} -> !tla.ptr<f32, l1, 512>
    %ub_ptr = tla.alloc_ptr{size_bytes = 4096} -> !tla.ptr<f32, ub, 256>
    %l1 = "tla.make_tensor_like"(%l1_ptr, %gm) {layoutTag = #tla.layout_tag<zN>} : (!tla.ptr<f32, l1, 512>, !gm) -> !l1
    %ub = "tla.make_tensor_like"(%ub_ptr, %gm) {layoutTag = #tla.layout_tag<RowMajor>} : (!tla.ptr<f32, ub, 256>, !gm) -> !ub
    "tla.cube"() ({
      tla.copy %l1, %gm : !l1, !gm
    }) : () -> ()
    "tla.vector"() ({
      tla.copy %ub, %gm : !ub, !gm
    }) : () -> ()
    tla.return
  }
}

// CHECK-LABEL: func.func @core_ratio_default
// CHECK-SAME: tla.core_ratio = array<i64: 1, 2>
