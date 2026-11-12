// RUN: %tla_compile %s -o - | %filecheck %s --check-prefix=CHECK-SPLITM
// RUN: sed -e 's/SPLIT_M/SPLIT_N/g' -e 's/splitm/splitn/g' -e 's/copy_l0c2ub_splitm/copy_l0c2ub_splitn/g' %s | %tla_compile - -o - | %filecheck %s --check-prefix=CHECK-SPLITN
//
// Test lowering of tla.copy from L0C to UB with CopyL0C2DstParams (SPLIT_M and SPLIT_N modes).
// Both use full-size (32,32) UB tensor matching L0C — splitting is handled at hardware level.

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
      %10 = tla.make_stride -> !tla.stride<32,1>
      %11 = tla.make_layout %9, %10 : !tla.shape<32,32>, !tla.stride<32,1> -> !tla.layout<!tla.shape<32,32>, !tla.stride<32,1>, !tla.shape<32,32>, RowMajor>
      %12 = tla.make_coord -> !tla.coord<0,0>
      %13 = tla.make_tensor %8, %11, %12 : !tla.ptr<f32, ub, 256>, !tla.layout<!tla.shape<32,32>, !tla.stride<32,1>, !tla.shape<32,32>, RowMajor>, !tla.coord<0,0> -> !tla.tensor<!tla.layout<!tla.shape<32,32>, !tla.stride<32,1>, !tla.shape<32,32>, RowMajor>, !tla.coord<0,0>, !tla.ptr<f32, ub, 256>>
      %14 = "tla.CopyL0C2DstParams"() <{unit_flag = 0 : i64, relu_enable = false, quant_mode = #tla.quant_mode<NO_QUANT>, l0c2ub_mode = #tla.l0c2ub_mode<SPLIT_M>}> : () -> !tla.copy_l0c2dst_params
      "tla.copy"(%13, %6, %14) : (!tla.tensor<!tla.layout<!tla.shape<32,32>, !tla.stride<32,1>, !tla.shape<32,32>, RowMajor>, !tla.coord<0,0>, !tla.ptr<f32, ub, 256>>, !tla.tensor<!tla.layout<!tla.shape<(16,2),(16,2)>, !tla.stride<(16,256),(1,512)>, !tla.shape<32,32>, L0Clayout>, !tla.coord<0,0>, !tla.ptr<f32, l0c, 1024>>, !tla.copy_l0c2dst_params) -> ()
    }) : () -> ()
    "tla.return"() : () -> ()
  }) {tla.exec_units = "cube", function_type = (!tla.tensor<!tla.layout<!tla.shape<32,32>, !tla.stride<32,1>, !tla.shape<32,32>, RowMajor>, !tla.coord<0,0>, !tla.ptr<f32, gm, 4>>) -> (), sym_name = "copy_l0c2ub_splitm"} : () -> ()
}

// ---- SPLIT_M ----
// CHECK-SPLITM: func.func private @copy_l0c_to_ub_RowMajor_splitm_float_float
// CHECK-SPLITM-SAME: hacc.always_inline
// CHECK-SPLITM-SAME: hivm.func_core_type = #hivm.func_core_type<AIC>
// CHECK-SPLITM-SAME: llvm.emit_c_interface
// CHECK-SPLITM-LABEL: func.func @copy_l0c2ub_splitm
// CHECK-SPLITM-DAG: [[CC:%.*]] = memref.cast {{%.*}} : memref<1024xf32, #hivm.address_space<cc>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<cc>>
// CHECK-SPLITM-DAG: [[UB:%.*]] = memref.cast {{%.*}} : memref<512xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
// CHECK-SPLITM: call @copy_l0c_to_ub_RowMajor_splitm_float_float([[CC]], [[UB]]
// CHECK-SPLITM-SAME: i8
// CHECK-SPLITM-SAME: i8
// CHECK-SPLITM-NOT: "tla.copy"
// CHECK-SPLITM-NOT: "tla.CopyL0C2DstParams"

// ---- SPLIT_N ----
// CHECK-SPLITN: func.func private @copy_l0c_to_ub_RowMajor_splitn_float_float
// CHECK-SPLITN-SAME: hacc.always_inline
// CHECK-SPLITN-SAME: hivm.func_core_type = #hivm.func_core_type<AIC>
// CHECK-SPLITN-SAME: llvm.emit_c_interface
// CHECK-SPLITN-LABEL: func.func @copy_l0c2ub_splitn
// CHECK-SPLITN-DAG: [[CC:%.*]] = memref.cast {{%.*}} : memref<1024xf32, #hivm.address_space<cc>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<cc>>
// CHECK-SPLITN-DAG: [[UB:%.*]] = memref.cast {{%.*}} : memref<512xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
// CHECK-SPLITN: call @copy_l0c_to_ub_RowMajor_splitn_float_float([[CC]], [[UB]]
// CHECK-SPLITN-SAME: i8
// CHECK-SPLITN-SAME: i8
// CHECK-SPLITN-NOT: "tla.copy"
// CHECK-SPLITN-NOT: "tla.CopyL0C2DstParams"
