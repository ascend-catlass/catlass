// RUN: %tla_compile %s -o - | %filecheck %s
// RUN: sed 's/ptr<f32/ptr<f8E4M3FN/g' %s | not %tla_compile - -o - 2>&1 | %filecheck %s --check-prefix=FP8-PAD-ERR
//
// GM→L1 VectorLayout DataCopyPad via CopyPadParams (cube):
//   %p = tla.CopyPadParams left, right, value_bits
//   tla.copy %l1, %gm, %p
// FP8+Pad must fail verify (BC would silently drop pad).

module {
  "tla.func"() ({
  ^bb0(%arg0: !tla.tensor<!tla.layout<!tla.shape<48>, !tla.stride<1>, !tla.shape<48>, VectorLayout>, !tla.coord<0>, !tla.ptr<f32, gm, 4>>):
    %l1_raw = tla.alloc_ptr{size_bytes = 256} -> !tla.ptr<i8, l1, 256>
    %l1_ptr = tla.recast_ptr %l1_raw : !tla.ptr<i8, l1, 256> -> !tla.ptr<f32, l1, 256>
    %shape = tla.make_shape -> !tla.shape<56>
    %stride = tla.make_stride -> !tla.stride<1>
    %layout = tla.make_layout %shape, %stride : !tla.shape<56>, !tla.stride<1> -> !tla.layout<!tla.shape<56>, !tla.stride<1>, !tla.shape<56>, VectorLayout>
    %coord = tla.make_coord -> !tla.coord<0>
    %l1 = tla.make_tensor %l1_ptr, %layout, %coord : !tla.ptr<f32, l1, 256>, !tla.layout<!tla.shape<56>, !tla.stride<1>, !tla.shape<56>, VectorLayout>, !tla.coord<0> -> !tla.tensor<!tla.layout<!tla.shape<56>, !tla.stride<1>, !tla.shape<56>, VectorLayout>, !tla.coord<0>, !tla.ptr<f32, l1, 256>>

    %c0 = arith.constant 0 : i64
    %c1 = arith.constant 1 : i64
    %pad = tla.CopyPadParams %c0, %c1, %c0 : i64, i64, i64
    "tla.cube"() ({
      tla.copy %l1, %arg0, %pad : !tla.tensor<!tla.layout<!tla.shape<56>, !tla.stride<1>, !tla.shape<56>, VectorLayout>, !tla.coord<0>, !tla.ptr<f32, l1, 256>>, !tla.tensor<!tla.layout<!tla.shape<48>, !tla.stride<1>, !tla.shape<48>, VectorLayout>, !tla.coord<0>, !tla.ptr<f32, gm, 4>>, !tla.copy_pad_params
    }) : () -> ()
    "tla.return"() : () -> ()
  }) {function_type = (!tla.tensor<!tla.layout<!tla.shape<48>, !tla.stride<1>, !tla.shape<48>, VectorLayout>, !tla.coord<0>, !tla.ptr<f32, gm, 4>>) -> (), sym_name = "copy_gm_l1_vl_pad"} : () -> ()
}

// CHECK: func.func private @copy_gm_VectorLayout_to_l1_VectorLayout_float
// CHECK-SAME: hivm.func_core_type = #hivm.func_core_type<AIC>
// CHECK-LABEL: func.func @copy_gm_l1_vl_pad
// Trailing ABI padEnable/left/right/valueBits asserted in
// test_gm2l1_vl_pad_lowering_trailing_i64s (pytest).
// CHECK: call @copy_gm_VectorLayout_to_l1_VectorLayout_float
// CHECK-NOT: "tla.copy"
// CHECK-NOT: "tla.CopyPadParams"
// FP8-PAD-ERR: copy_pad_params on GM{{.*}}L1 does not support fp8/fp4
