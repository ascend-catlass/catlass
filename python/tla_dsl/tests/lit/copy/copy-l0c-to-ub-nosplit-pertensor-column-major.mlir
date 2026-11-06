// RUN: %tla_compile %s -o - | %filecheck %s
//
// Test lowering of tla.copy from L0C to UB with a ColumnMajor destination in
// PER_TENSOR (on-the-fly) quant mode (NO_SPLIT_VEC_0). The scalar scale is
// carried as an extra f32 operand, so the call folds into the `_pertensor_`
// runtime symbol together with the `subBlockId`. f32 accumulator -> f16 UB.

module attributes {tla.module_exec_units = "cube"} {
  "tla.func"() ({
  ^bb0(%arg0: !tla.tensor<!tla.layout<!tla.shape<32,32>, !tla.stride<32,1>, !tla.shape<32,32>, RowMajor>, !tla.coord<0,0>, !tla.ptr<f32, gm, 4>>):
    %0 = tla.make_shape -> !tla.shape<32,32>
    %1 = tla.make_coord -> !tla.coord<0,0>
    %2 = tla.make_coord -> !tla.coord<0,0>
    %3 = tla.tile_view %arg0, %0, %2 : !tla.tensor<!tla.layout<!tla.shape<32,32>, !tla.stride<32,1>, !tla.shape<32,32>, RowMajor>, !tla.coord<0,0>, !tla.ptr<f32, gm, 4>>, !tla.shape<32,32>, !tla.coord<0,0> -> !tla.tensor<!tla.layout<!tla.shape<32,32>, !tla.stride<32,1>, !tla.shape<32,32>, RowMajor>, !tla.coord<0,0>, !tla.ptr<f32, gm, 4>>

    %4 = tla.alloc_ptr{size_bytes = 4096} -> !tla.ptr<i8, l0c, 512>
    %5 = tla.recast_ptr %4 : !tla.ptr<i8, l0c, 512> -> !tla.ptr<f32, l0c, 512>
    %6 = tla.make_tensor_like %5 like %3 layoutTag(#tla.layout_tag<L0Clayout>) : !tla.ptr<f32, l0c, 512>, !tla.tensor<!tla.layout<!tla.shape<32,32>, !tla.stride<32,1>, !tla.shape<32,32>, RowMajor>, !tla.coord<0,0>, !tla.ptr<f32, gm, 4>> -> !tla.tensor<!tla.layout<!tla.shape<(16,2),(16,2)>, !tla.stride<(16,256),(1,512)>, !tla.shape<32,32>, L0Clayout>, !tla.coord<0,0>, !tla.ptr<f32, l0c, 512>>

    %7 = tla.alloc_ptr{size_bytes = 2048} -> !tla.ptr<i8, ub, 256>
    %8 = tla.recast_ptr %7 : !tla.ptr<i8, ub, 256> -> !tla.ptr<f16, ub, 256>

    "tla.cube"() ({
      %9 = tla.make_shape -> !tla.shape<32,32>
      %10 = tla.make_stride -> !tla.stride<1,32>
      %11 = tla.make_layout %9, %10 : !tla.shape<32,32>, !tla.stride<1,32> -> !tla.layout<!tla.shape<32,32>, !tla.stride<1,32>, !tla.shape<32,32>, ColumnMajor>
      %12 = tla.make_coord -> !tla.coord<0,0>
      %13 = tla.make_tensor %8, %11, %12 : !tla.ptr<f16, ub, 256>, !tla.layout<!tla.shape<32,32>, !tla.stride<1,32>, !tla.shape<32,32>, ColumnMajor>, !tla.coord<0,0> -> !tla.tensor<!tla.layout<!tla.shape<32,32>, !tla.stride<1,32>, !tla.shape<32,32>, ColumnMajor>, !tla.coord<0,0>, !tla.ptr<f16, ub, 256>>
      %scale = "arith.constant"() <{value = 1.000000e+00 : f32}> : () -> f32
      %off = "arith.constant"() <{value = 0 : i32}> : () -> i32
      %14 = "tla.CopyL0C2DstParams"(%scale, %off) <{unit_flag = 0 : i64, relu_enable = false, quant_mode = #tla.quant_mode<PER_TENSOR>, l0c2ub_mode = #tla.l0c2ub_mode<NO_SPLIT_VEC_0>, operandSegmentSizes = array<i32: 1, 1, 0>}> : (f32, i32) -> !tla.copy_l0c2dst_params
      "tla.copy"(%13, %6, %14) : (!tla.tensor<!tla.layout<!tla.shape<32,32>, !tla.stride<1,32>, !tla.shape<32,32>, ColumnMajor>, !tla.coord<0,0>, !tla.ptr<f16, ub, 256>>, !tla.tensor<!tla.layout<!tla.shape<(16,2),(16,2)>, !tla.stride<(16,256),(1,512)>, !tla.shape<32,32>, L0Clayout>, !tla.coord<0,0>, !tla.ptr<f32, l0c, 512>>, !tla.copy_l0c2dst_params) -> ()
    }) : () -> ()
    "tla.return"() : () -> ()
  }) {tla.exec_units = "cube", function_type = (!tla.tensor<!tla.layout<!tla.shape<32,32>, !tla.stride<32,1>, !tla.shape<32,32>, RowMajor>, !tla.coord<0,0>, !tla.ptr<f32, gm, 4>>) -> (), sym_name = "copy_l0c2ub_col_pertensor_f16"} : () -> ()
}

// CHECK: func.func private @copy_l0c_to_ub_ColumnMajor_nosplit_pertensor_float_half
// CHECK-SAME: hacc.always_inline
// CHECK-SAME: hivm.func_core_type = #hivm.func_core_type<AIC>
// CHECK-SAME: llvm.emit_c_interface
// CHECK-LABEL: func.func @copy_l0c2ub_col_pertensor_f16
// CHECK-DAG: [[CC:%.*]] = memref.cast {{%.*}} : memref<1024xf32, #hivm.address_space<cc>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<cc>>
// CHECK-DAG: [[UB:%.*]] = memref.cast {{%.*}} : memref<1024xf16, #hivm.address_space<ub>> to memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<ub>>
// CHECK: call @copy_l0c_to_ub_ColumnMajor_nosplit_pertensor_float_half([[CC]], [[UB]]
// CHECK-SAME: i8, i8, f32
// CHECK-NOT: "tla.copy"
// CHECK-NOT: "tla.CopyL0C2DstParams"
