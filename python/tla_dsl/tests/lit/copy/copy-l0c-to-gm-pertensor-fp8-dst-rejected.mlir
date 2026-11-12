// A per-tensor (PRE) quantized f32 accumulator narrows only to f16, bf16 and the
// 8-bit integer destinations. fp8 is not a quantized fixpipe destination: of the
// two fp8 encodings the on-the-fly quant supports e4m3fn only, and e5m2 has no
// quant route at all. The quant rows of the fixpipe route table in
// catlass/core_api.py carry no float8 entry, and no REGISTER_L0C_TO_* is
// instantiated for one.
//
// This has to be rejected here, in the route resolution, rather than relied upon
// to fail elsewhere: fp8 has a copyRuntimeElemSuffix mapping (the GM->L1,
// L1->L0 and GM<->UB routes need it), so an unguarded accumulator would resolve
// this copy to @copy_l0c_to_gm_RowMajor_pertensor_float_fp8_e5m2_t -- a
// well-formed name with no implementation behind it, which surfaces as a link
// failure instead of a diagnostic. The DSL frontend also blocks this, but the
// compiler must not depend on the frontend for IR it did not produce.
//
// e5m2 is the encoding to test here: it has no quant route in any quant mode, so
// this rejection holds even if e4m3fn is enabled later.

// RUN: not %tla_compile %s -o - 2>&1 | %filecheck %s --check-prefix=FP8DST
// FP8DST: tla.copy descriptor/layout combination is unsupported: l0c(L0Clayout) -> gm(RowMajor)

module attributes {tla.module_exec_units = "cube"} {
  "tla.func"() ({
  ^bb0(%arg0: !tla.tensor<!tla.layout<!tla.shape<32,32>, !tla.stride<32,1>, !tla.shape<32,32>, RowMajor>, !tla.coord<0,0>, !tla.ptr<f8E5M2, gm, 4>>):
    %0 = "tla.make_shape"() : () -> !tla.shape<32,32>
    %1 = "tla.make_coord"() : () -> !tla.coord<0,0>
    %2 = "tla.make_coord"() : () -> !tla.coord<0,0>
    %3 = "tla.tile_view"(%arg0, %0, %2) : (!tla.tensor<!tla.layout<!tla.shape<32,32>, !tla.stride<32,1>, !tla.shape<32,32>, RowMajor>, !tla.coord<0,0>, !tla.ptr<f8E5M2, gm, 4>>, !tla.shape<32,32>, !tla.coord<0,0>) -> !tla.tensor<!tla.layout<!tla.shape<32,32>, !tla.stride<32,1>, !tla.shape<32,32>, RowMajor>, !tla.coord<0,0>, !tla.ptr<f8E5M2, gm, 4>>

    %4 = "tla.alloc_ptr"() {size_bytes = 4096 : i64} : () -> !tla.ptr<i8, l0c, 1024>
    %5 = "tla.recast_ptr"(%4) : (!tla.ptr<i8, l0c, 1024>) -> !tla.ptr<f32, l0c, 1024>
    %6 = "tla.make_tensor_like"(%5, %3) {layoutTag = #tla.layout_tag<L0Clayout>} : (!tla.ptr<f32, l0c, 1024>, !tla.tensor<!tla.layout<!tla.shape<32,32>, !tla.stride<32,1>, !tla.shape<32,32>, RowMajor>, !tla.coord<0,0>, !tla.ptr<f8E5M2, gm, 4>>) -> !tla.tensor<!tla.layout<!tla.shape<(16,2),(16,2)>, !tla.stride<(16,256),(1,512)>, !tla.shape<32,32>, L0Clayout>, !tla.coord<0,0>, !tla.ptr<f32, l0c, 1024>>

    %scale = "arith.constant"() <{value = 1.000000e+00 : f32}> : () -> f32
    %off = "arith.constant"() <{value = 0 : i32}> : () -> i32
    %7 = "tla.CopyL0C2DstParams"(%scale, %off) <{unit_flag = 0 : i64, relu_enable = false, quant_mode = #tla.quant_mode<PER_TENSOR>, l0c2ub_mode = #tla.l0c2ub_mode<NO_SPLIT_VEC_0>, operandSegmentSizes = array<i32: 1, 1, 0>}> : (f32, i32) -> !tla.copy_l0c2dst_params

    "tla.cube"() ({
      "tla.copy"(%3, %6, %7) : (!tla.tensor<!tla.layout<!tla.shape<32,32>, !tla.stride<32,1>, !tla.shape<32,32>, RowMajor>, !tla.coord<0,0>, !tla.ptr<f8E5M2, gm, 4>>, !tla.tensor<!tla.layout<!tla.shape<(16,2),(16,2)>, !tla.stride<(16,256),(1,512)>, !tla.shape<32,32>, L0Clayout>, !tla.coord<0,0>, !tla.ptr<f32, l0c, 1024>>, !tla.copy_l0c2dst_params) -> ()
    }) : () -> ()
    "tla.return"() : () -> ()
  }) {tla.exec_units = "cube", function_type = (!tla.tensor<!tla.layout<!tla.shape<32,32>, !tla.stride<32,1>, !tla.shape<32,32>, RowMajor>, !tla.coord<0,0>, !tla.ptr<f8E5M2, gm, 4>>) -> (), sym_name = "copy_l0c2gm_pertensor_fp8_e5m2_rejected"} : () -> ()
}
