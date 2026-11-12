// An f32 L0C accumulator can only leave fixpipe as f32, f16 or bf16. Nothing
// converts it to fp8, and there is no REGISTER_L0C_TO_UB for an fp8 destination.
//
// This has to be rejected here, in the route resolution, rather than relied upon
// to fail elsewhere: fp8 has a copyRuntimeElemSuffix mapping (it needs one for
// the GM->L1 and L1->L0 routes), so an unguarded f32 accumulator would resolve
// this copy to @copy_l0c_to_ub_RowMajor_nosplit_float_fp8_e4m3fn_t -- a well-formed name
// with no implementation behind it, which surfaces as a link failure instead of
// a diagnostic. The DSL frontend also blocks this, but the compiler must not
// depend on the frontend for IR it did not produce.

// RUN: not %tla_compile %s -o - 2>&1 | %filecheck %s --check-prefix=FP8DST
// FP8DST: tla.copy descriptor/layout combination is unsupported: l0c(L0Clayout) -> ub(RowMajor)

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

    "tla.cube"() ({
      %7 = tla.alloc_ptr{size_bytes = 4096} -> !tla.ptr<i8, ub, 256>
      %8 = tla.recast_ptr %7 : !tla.ptr<i8, ub, 256> -> !tla.ptr<f8E4M3FN, ub, 256>
      %9 = tla.make_shape -> !tla.shape<32,32>
      %10 = tla.make_stride -> !tla.stride<32,1>
      %11 = tla.make_layout %9, %10 : !tla.shape<32,32>, !tla.stride<32,1> -> !tla.layout<!tla.shape<32,32>, !tla.stride<32,1>, !tla.shape<32,32>, RowMajor>
      %12 = tla.make_coord -> !tla.coord<0,0>
      %13 = tla.make_tensor %8, %11, %12 : !tla.ptr<f8E4M3FN, ub, 256>, !tla.layout<!tla.shape<32,32>, !tla.stride<32,1>, !tla.shape<32,32>, RowMajor>, !tla.coord<0,0> -> !tla.tensor<!tla.layout<!tla.shape<32,32>, !tla.stride<32,1>, !tla.shape<32,32>, RowMajor>, !tla.coord<0,0>, !tla.ptr<f8E4M3FN, ub, 256>>
      %14 = "tla.CopyL0C2DstParams"() <{unit_flag = 0 : i64, relu_enable = false, quant_mode = #tla.quant_mode<NO_QUANT>, l0c2ub_mode = #tla.l0c2ub_mode<NO_SPLIT_VEC_1>}> : () -> !tla.copy_l0c2dst_params

      "tla.copy"(%13, %6, %14) : (!tla.tensor<!tla.layout<!tla.shape<32,32>, !tla.stride<32,1>, !tla.shape<32,32>, RowMajor>, !tla.coord<0,0>, !tla.ptr<f8E4M3FN, ub, 256>>, !tla.tensor<!tla.layout<!tla.shape<(16,2),(16,2)>, !tla.stride<(16,256),(1,512)>, !tla.shape<32,32>, L0Clayout>, !tla.coord<0,0>, !tla.ptr<f32, l0c, 1024>>, !tla.copy_l0c2dst_params) -> ()
    }) : () -> ()
    "tla.return"() : () -> ()
  }) {tla.exec_units = "cube", function_type = (!tla.tensor<!tla.layout<!tla.shape<32,32>, !tla.stride<32,1>, !tla.shape<32,32>, RowMajor>, !tla.coord<0,0>, !tla.ptr<f32, gm, 4>>) -> (), sym_name = "copy_l0c2ub_nosplit"} : () -> ()
}
