// RUN: %tla_compile %s -o - | %filecheck %s --check-prefix=CHECK-F32
// RUN: sed -e 's/f32, ub/f16, ub/g' -e 's/copy_l0c2ub_col_f32/copy_l0c2ub_col_f16/g' %s | %tla_compile - -o - | %filecheck %s --check-prefix=CHECK-F16
//
// Test lowering of tla.copy from L0C to UB with ColumnMajor destination (NO_SPLIT_VEC_0).
// F32->F32 and F32->F16 copy routes.

module attributes {tla.module_exec_units = "cube"} {
  "tla.func"() ({
  ^bb0(%arg0: !tla.tensor<!tla.layout<!tla.shape<32,32>, !tla.stride<32,1>, !tla.shape<32,32>, RowMajor>, !tla.coord<0,0>, !tla.ptr<f32, gm, 4>>):
    %0 = tla.make_shape -> !tla.shape<32,32>
    %1 = tla.make_coord -> !tla.coord<0,0>
    %2 = tla.make_coord -> !tla.coord<0,0>
    %3 = tla.tile_view %arg0, %0, %2 : !tla.tensor<!tla.layout<!tla.shape<32,32>, !tla.stride<32,1>, !tla.shape<32,32>, RowMajor>, !tla.coord<0,0>, !tla.ptr<f32, gm, 4>>, !tla.shape<32,32>, !tla.coord<0,0> -> !tla.tensor<!tla.layout<!tla.shape<32,32>, !tla.stride<32,1>, !tla.shape<32,32>, RowMajor>, !tla.coord<0,0>, !tla.ptr<f32, gm, 4>>

    %4 = tla.alloc_ptr{size_bytes = 4096} -> !tla.ptr<i8, l0c, 1024>
    %5 = tla.recast_ptr %4 : !tla.ptr<i8, l0c, 1024> -> !tla.ptr<f32, l0c, 1024>
    %6 = tla.make_tensor_like %5 like %3 layoutTag(#tla.layout_tag<L0Clayout>) : !tla.ptr<f32, l0c, 1024>, !tla.tensor<!tla.layout<!tla.shape<32,32>, !tla.stride<32,1>, !tla.shape<32,32>, RowMajor>, !tla.coord<0,0>, !tla.ptr<f32, gm, 4>> -> !tla.tensor<!tla.layout<!tla.shape<(16,2),(16,2)>, !tla.stride<(16,256),(1,512)>, !tla.shape<32,32>, L0Clayout>, !tla.coord<0,0>, !tla.ptr<f32, l0c, 1024>>

    %7 = tla.alloc_ptr{size_bytes = 2048} -> !tla.ptr<i8, ub, 256>
    %8 = tla.recast_ptr %7 : !tla.ptr<i8, ub, 256> -> !tla.ptr<f32, ub, 256>

    "tla.cube"() ({
      %9 = tla.make_shape -> !tla.shape<32,32>
      %10 = tla.make_stride -> !tla.stride<1,32>
      %11 = tla.make_layout %9, %10 : !tla.shape<32,32>, !tla.stride<1,32> -> !tla.layout<!tla.shape<32,32>, !tla.stride<1,32>, !tla.shape<32,32>, ColumnMajor>
      %12 = tla.make_coord -> !tla.coord<0,0>
      %13 = tla.make_tensor %8, %11, %12 : !tla.ptr<f32, ub, 256>, !tla.layout<!tla.shape<32,32>, !tla.stride<1,32>, !tla.shape<32,32>, ColumnMajor>, !tla.coord<0,0> -> !tla.tensor<!tla.layout<!tla.shape<32,32>, !tla.stride<1,32>, !tla.shape<32,32>, ColumnMajor>, !tla.coord<0,0>, !tla.ptr<f32, ub, 256>>
      %14 = "tla.CopyL0C2DstParams"() <{unit_flag = 0 : i64, relu_enable = false, quant_mode = #tla.quant_mode<NO_QUANT>, l0c2ub_mode = #tla.l0c2ub_mode<NO_SPLIT_VEC_0>}> : () -> !tla.copy_l0c2dst_params
      "tla.copy"(%13, %6, %14) : (!tla.tensor<!tla.layout<!tla.shape<32,32>, !tla.stride<1,32>, !tla.shape<32,32>, ColumnMajor>, !tla.coord<0,0>, !tla.ptr<f32, ub, 256>>, !tla.tensor<!tla.layout<!tla.shape<(16,2),(16,2)>, !tla.stride<(16,256),(1,512)>, !tla.shape<32,32>, L0Clayout>, !tla.coord<0,0>, !tla.ptr<f32, l0c, 1024>>, !tla.copy_l0c2dst_params) -> ()
    }) : () -> ()
    "tla.return"() : () -> ()
  }) {tla.exec_units = "cube", function_type = (!tla.tensor<!tla.layout<!tla.shape<32,32>, !tla.stride<32,1>, !tla.shape<32,32>, RowMajor>, !tla.coord<0,0>, !tla.ptr<f32, gm, 4>>) -> (), sym_name = "copy_l0c2ub_col_f32"} : () -> ()
}

// ---- F32->F32 ----
// CHECK-F32: func.func private @copy_l0c_to_ub_ColumnMajor_nosplit_float_float
// CHECK-F32-SAME: hacc.always_inline
// CHECK-F32-SAME: hivm.func_core_type = #hivm.func_core_type<AIC>
// CHECK-F32-SAME: llvm.emit_c_interface
// CHECK-F32-LABEL: func.func @copy_l0c2ub_col_f32
// CHECK-F32-DAG: [[CC:%.*]] = memref.cast {{%.*}} : memref<1024xf32, #hivm.address_space<cc>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<cc>>
// CHECK-F32-DAG: [[UB:%.*]] = memref.cast {{%.*}} : memref<512xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
// CHECK-F32: call @copy_l0c_to_ub_ColumnMajor_nosplit_float_float([[CC]], [[UB]]
// CHECK-F32-NOT: "tla.copy"
// CHECK-F32-NOT: "tla.CopyL0C2DstParams"

// ---- F32->F16 ----
// CHECK-F16: func.func private @copy_l0c_to_ub_ColumnMajor_nosplit_float_half
// CHECK-F16-SAME: hacc.always_inline
// CHECK-F16-SAME: hivm.func_core_type = #hivm.func_core_type<AIC>
// CHECK-F16-SAME: llvm.emit_c_interface
// CHECK-F16-LABEL: func.func @copy_l0c2ub_col_f16
// CHECK-F16-DAG: [[CC:%.*]] = memref.cast {{%.*}} : memref<1024xf32, #hivm.address_space<cc>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<cc>>
// CHECK-F16-DAG: [[UB:%.*]] = memref.cast {{%.*}} : memref<1024xf16, #hivm.address_space<ub>> to memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<ub>>
// CHECK-F16: call @copy_l0c_to_ub_ColumnMajor_nosplit_float_half([[CC]], [[UB]]
// CHECK-F16-NOT: "tla.copy"
// CHECK-F16-NOT: "tla.CopyL0C2DstParams"
