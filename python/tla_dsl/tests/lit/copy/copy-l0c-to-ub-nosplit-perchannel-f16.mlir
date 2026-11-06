// RUN: %tla_compile %s -o - | %filecheck %s
//
// Test lowering of tla.copy from L0C to UB with a RowMajor destination in
// PER_CHANNEL (on-the-fly) quant mode (NO_SPLIT_VEC_0). The per-channel scale is
// a tensor: one packed uint64 word per output channel, staged in L1 as a flat
// VectorLayout tile and passed as the dequant operand, so the call folds into
// the `_perchannel_` runtime symbol together with the quant tensor's memref.
// f32 accumulator -> f16 UB.

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

    // The per-channel scale tile: 32 packed words, one row, flat VectorLayout.
    %9 = tla.alloc_ptr{size_bytes = 256} -> !tla.ptr<i8, l1, 512>
    %10 = tla.recast_ptr %9 : !tla.ptr<i8, l1, 512> -> !tla.ptr<ui64, l1, 512>
    %11 = tla.make_shape -> !tla.shape<32>
    %12 = tla.make_stride -> !tla.stride<1>
    %13 = tla.make_layout %11, %12 {layoutTag = #tla.layout_tag<VectorLayout>} : !tla.shape<32>, !tla.stride<1> -> !tla.layout<!tla.shape<32>, !tla.stride<1>, !tla.shape<32>, VectorLayout>
    %14 = tla.make_coord -> !tla.coord<0>
    %15 = tla.make_tensor %10, %13, %14 : !tla.ptr<ui64, l1, 512>, !tla.layout<!tla.shape<32>, !tla.stride<1>, !tla.shape<32>, VectorLayout>, !tla.coord<0> -> !tla.tensor<!tla.layout<!tla.shape<32>, !tla.stride<1>, !tla.shape<32>, VectorLayout>, !tla.coord<0>, !tla.ptr<ui64, l1, 512>>

    "tla.cube"() ({
      %16 = tla.make_shape -> !tla.shape<32,32>
      %17 = tla.make_stride -> !tla.stride<32,1>
      %18 = tla.make_layout %16, %17 : !tla.shape<32,32>, !tla.stride<32,1> -> !tla.layout<!tla.shape<32,32>, !tla.stride<32,1>, !tla.shape<32,32>, RowMajor>
      %19 = tla.make_coord -> !tla.coord<0,0>
      %20 = tla.make_tensor %8, %18, %19 : !tla.ptr<f16, ub, 256>, !tla.layout<!tla.shape<32,32>, !tla.stride<32,1>, !tla.shape<32,32>, RowMajor>, !tla.coord<0,0> -> !tla.tensor<!tla.layout<!tla.shape<32,32>, !tla.stride<32,1>, !tla.shape<32,32>, RowMajor>, !tla.coord<0,0>, !tla.ptr<f16, ub, 256>>
      %21 = "tla.CopyL0C2DstParams"(%15) <{unit_flag = 0 : i64, relu_enable = false, quant_mode = #tla.quant_mode<PER_CHANNEL>, l0c2ub_mode = #tla.l0c2ub_mode<NO_SPLIT_VEC_0>, operandSegmentSizes = array<i32: 0, 0, 1>}> : (!tla.tensor<!tla.layout<!tla.shape<32>, !tla.stride<1>, !tla.shape<32>, VectorLayout>, !tla.coord<0>, !tla.ptr<ui64, l1, 512>>) -> !tla.copy_l0c2dst_params
      "tla.copy"(%20, %6, %21) : (!tla.tensor<!tla.layout<!tla.shape<32,32>, !tla.stride<32,1>, !tla.shape<32,32>, RowMajor>, !tla.coord<0,0>, !tla.ptr<f16, ub, 256>>, !tla.tensor<!tla.layout<!tla.shape<(16,2),(16,2)>, !tla.stride<(16,256),(1,512)>, !tla.shape<32,32>, L0Clayout>, !tla.coord<0,0>, !tla.ptr<f32, l0c, 512>>, !tla.copy_l0c2dst_params) -> ()
    }) : () -> ()
    "tla.return"() : () -> ()
  }) {tla.exec_units = "cube", function_type = (!tla.tensor<!tla.layout<!tla.shape<32,32>, !tla.stride<32,1>, !tla.shape<32,32>, RowMajor>, !tla.coord<0,0>, !tla.ptr<f32, gm, 4>>) -> (), sym_name = "copy_l0c2ub_row_perchannel_f16"} : () -> ()
}

// CHECK: func.func private @copy_l0c_to_ub_RowMajor_nosplit_perchannel_float_half
// CHECK-SAME: hacc.always_inline
// CHECK-SAME: hivm.func_core_type = #hivm.func_core_type<AIC>
// CHECK-SAME: llvm.emit_c_interface
// CHECK-LABEL: func.func @copy_l0c2ub_row_perchannel_f16
// CHECK-DAG: [[CC:%.*]] = memref.cast {{%.*}} : memref<1024xf32, #hivm.address_space<cc>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<cc>>
// CHECK-DAG: [[UB:%.*]] = memref.cast {{%.*}} : memref<1024xf16, #hivm.address_space<ub>> to memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<ub>>
// CHECK-DAG: [[Q:%.*]] = memref.cast {{%.*}} : memref<32xui64, #hivm.address_space<cbuf>> to memref<?xui64, strided<[?], offset: ?>, #hivm.address_space<cbuf>>
// CHECK: call @copy_l0c_to_ub_RowMajor_nosplit_perchannel_float_half([[CC]], [[UB]]
// CHECK-SAME: {{.*}}[[Q]]
// CHECK-NOT: "tla.copy"
// CHECK-NOT: "tla.CopyL0C2DstParams"
