// RUN: %tla_compile %s -o - | %filecheck %s
//
// Test lowering of tla.copy from L0C to GM with atomic add operation.
// The copy should be enclosed by the set_ctrl sequence for atomic add f32,
// followed by the set_ctrl sequence that disables atomic writes.

module attributes {tla.module_exec_units = "cube"} {
  "tla.func"() ({
  ^bb0(%arg0: !tla.tensor<!tla.layout<!tla.shape<32,32>, !tla.stride<32,1>, !tla.shape<32,32>, RowMajor>, !tla.coord<0,0>, !tla.ptr<f32, gm, 4>>):
    %0 = "tla.make_shape"() : () -> !tla.shape<32,32>
    %1 = "tla.make_coord"() : () -> !tla.coord<0,0>
    %2 = "tla.make_coord"() : () -> !tla.coord<0,0>
    %3 = "tla.tile_view"(%arg0, %0, %2) : (!tla.tensor<!tla.layout<!tla.shape<32,32>, !tla.stride<32,1>, !tla.shape<32,32>, RowMajor>, !tla.coord<0,0>, !tla.ptr<f32, gm, 4>>, !tla.shape<32,32>, !tla.coord<0,0>) -> !tla.tensor<!tla.layout<!tla.shape<32,32>, !tla.stride<32,1>, !tla.shape<32,32>, RowMajor>, !tla.coord<0,0>, !tla.ptr<f32, gm, 4>>

    %4 = "tla.alloc_ptr"() {size_bytes = 4096 : i64} : () -> !tla.ptr<i8, l0c, 1024>
    %5 = "tla.recast_ptr"(%4) : (!tla.ptr<i8, l0c, 1024>) -> !tla.ptr<f32, l0c, 1024>
    %6 = "tla.make_tensor_like"(%5, %3) {layoutTag = #tla.layout_tag<L0Clayout>} : (!tla.ptr<f32, l0c, 1024>, !tla.tensor<!tla.layout<!tla.shape<32,32>, !tla.stride<32,1>, !tla.shape<32,32>, RowMajor>, !tla.coord<0,0>, !tla.ptr<f32, gm, 4>>) -> !tla.tensor<!tla.layout<!tla.shape<(16,2),(16,2)>, !tla.stride<(16,256),(1,512)>, !tla.shape<32,32>, L0Clayout>, !tla.coord<0,0>, !tla.ptr<f32, l0c, 1024>>

    %7 = "tla.CopyL0C2DstParams"() <{unit_flag = 3 : i64, relu_enable = false, quant_mode = #tla.quant_mode<NO_QUANT>, l0c2ub_mode = #tla.l0c2ub_mode<NO_SPLIT_VEC_0>}> : () -> !tla.copy_l0c2dst_params

    "tla.cube"() ({
      "tla.copy"(%3, %6, %7) {atomic_mode = #tla.atomic_mode<add>} : (!tla.tensor<!tla.layout<!tla.shape<32,32>, !tla.stride<32,1>, !tla.shape<32,32>, RowMajor>, !tla.coord<0,0>, !tla.ptr<f32, gm, 4>>, !tla.tensor<!tla.layout<!tla.shape<(16,2),(16,2)>, !tla.stride<(16,256),(1,512)>, !tla.shape<32,32>, L0Clayout>, !tla.coord<0,0>, !tla.ptr<f32, l0c, 1024>>, !tla.copy_l0c2dst_params) -> ()
    }) : () -> ()
    "tla.return"() : () -> ()
  }) {tla.exec_units = "cube", function_type = (!tla.tensor<!tla.layout<!tla.shape<32,32>, !tla.stride<32,1>, !tla.shape<32,32>, RowMajor>, !tla.coord<0,0>, !tla.ptr<f32, gm, 4>>) -> (), sym_name = "copy_l0c2gm_atomic"} : () -> ()
}

// CHECK-LABEL: func.func @copy_l0c2gm_atomic
// CHECK: hivm.hir.set_ctrl true at ctrl[6]
// CHECK-NEXT: hivm.hir.set_ctrl false at ctrl[7]
// CHECK-NEXT: hivm.hir.set_ctrl false at ctrl[8]
// CHECK-NEXT: hivm.hir.set_ctrl false at ctrl[9]
// CHECK-NEXT: hivm.hir.set_ctrl false at ctrl[10]
// CHECK: call @copy_l0c_to_gm_RowMajor_float_float
// CHECK: hivm.hir.set_ctrl false at ctrl[6]
// CHECK-NEXT: hivm.hir.set_ctrl false at ctrl[7]
// CHECK-NEXT: hivm.hir.set_ctrl false at ctrl[8]
// CHECK-NOT: "tla.copy"
// CHECK-NOT: "tla.CopyL0C2DstParams"
