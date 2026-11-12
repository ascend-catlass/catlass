// RUN: %tla_compile %s -o - | %filecheck %s
//
// Test lowering of tla.copy from L0C to GM with CopyL0C2DstParams in
// PER_TENSOR (on-the-fly) quant + ReLU mode: one scalar scale is applied to the whole
// tile on the way out. f32 accumulator -> f16 GM.

module attributes {tla.module_exec_units = "cube"} {
  "tla.func"() ({
  ^bb0(%arg0: !tla.tensor<!tla.layout<!tla.shape<32,32>, !tla.stride<32,1>, !tla.shape<32,32>, RowMajor>, !tla.coord<0,0>, !tla.ptr<f16, gm, 4>>):
    %0 = "tla.make_shape"() : () -> !tla.shape<32,32>
    %1 = "tla.make_coord"() : () -> !tla.coord<0,0>
    %2 = "tla.make_coord"() : () -> !tla.coord<0,0>
    %3 = "tla.tile_view"(%arg0, %0, %2) : (!tla.tensor<!tla.layout<!tla.shape<32,32>, !tla.stride<32,1>, !tla.shape<32,32>, RowMajor>, !tla.coord<0,0>, !tla.ptr<f16, gm, 4>>, !tla.shape<32,32>, !tla.coord<0,0>) -> !tla.tensor<!tla.layout<!tla.shape<32,32>, !tla.stride<32,1>, !tla.shape<32,32>, RowMajor>, !tla.coord<0,0>, !tla.ptr<f16, gm, 4>>

    %4 = "tla.alloc_ptr"() {size_bytes = 4096 : i64} : () -> !tla.ptr<i8, l0c, 1024>
    %5 = "tla.recast_ptr"(%4) : (!tla.ptr<i8, l0c, 1024>) -> !tla.ptr<f32, l0c, 1024>
    %6 = "tla.make_tensor_like"(%5, %3) {layoutTag = #tla.layout_tag<L0Clayout>} : (!tla.ptr<f32, l0c, 1024>, !tla.tensor<!tla.layout<!tla.shape<32,32>, !tla.stride<32,1>, !tla.shape<32,32>, RowMajor>, !tla.coord<0,0>, !tla.ptr<f16, gm, 4>>) -> !tla.tensor<!tla.layout<!tla.shape<(16,2),(16,2)>, !tla.stride<(16,256),(1,512)>, !tla.shape<32,32>, L0Clayout>, !tla.coord<0,0>, !tla.ptr<f32, l0c, 1024>>

    %scale = "arith.constant"() <{value = 1.000000e+00 : f32}> : () -> f32
      %off = "arith.constant"() <{value = 0 : i32}> : () -> i32
    %7 = "tla.CopyL0C2DstParams"(%scale, %off) <{unit_flag = 0 : i64, relu_enable = true, quant_mode = #tla.quant_mode<PER_TENSOR>, l0c2ub_mode = #tla.l0c2ub_mode<NO_SPLIT_VEC_0>, operandSegmentSizes = array<i32: 1, 1, 0>}> : (f32, i32) -> !tla.copy_l0c2dst_params

    "tla.cube"() ({
      "tla.copy"(%3, %6, %7) : (!tla.tensor<!tla.layout<!tla.shape<32,32>, !tla.stride<32,1>, !tla.shape<32,32>, RowMajor>, !tla.coord<0,0>, !tla.ptr<f16, gm, 4>>, !tla.tensor<!tla.layout<!tla.shape<(16,2),(16,2)>, !tla.stride<(16,256),(1,512)>, !tla.shape<32,32>, L0Clayout>, !tla.coord<0,0>, !tla.ptr<f32, l0c, 1024>>, !tla.copy_l0c2dst_params) -> ()
    }) : () -> ()
    "tla.return"() : () -> ()
  }) {tla.exec_units = "cube", function_type = (!tla.tensor<!tla.layout<!tla.shape<32,32>, !tla.stride<32,1>, !tla.shape<32,32>, RowMajor>, !tla.coord<0,0>, !tla.ptr<f16, gm, 4>>) -> (), sym_name = "copy_l0c2gm_pertensor_relu_f16"} : () -> ()
}

// The PER_TENSOR fixpipe meta-op takes the scale as an extra f32 operand and
// folds into the `_pertensor_` runtime symbol.
// CHECK: func.func private @copy_l0c_to_gm_RowMajor_pertensor_relu_float_half
// CHECK-LABEL: func.func @copy_l0c2gm_pertensor_relu_f16
// CHECK: call @copy_l0c_to_gm_RowMajor_pertensor_relu_float_half
// CHECK-NOT: "tla.copy"
// CHECK-NOT: "tla.CopyL0C2DstParams"
