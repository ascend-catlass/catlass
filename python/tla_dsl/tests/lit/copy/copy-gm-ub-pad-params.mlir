// RUN: %tla_compile %s -o - | %filecheck %s
//
// GM→UB DataCopyPad via typed params (same pattern as CopyL0C2DstParams):
//   %p = tla.CopyPadParams left, right, value_bits
//   tla.copy %ub, %gm, %p

module {
  "tla.func"() ({
  ^bb0(%arg0: !tla.tensor<!tla.layout<!tla.shape<8,7>, !tla.stride<7,1>, !tla.shape<8,7>, RowMajor>, !tla.coord<0,0>, !tla.ptr<f32, gm, 4>>):
    %ub_raw = tla.alloc_ptr{size_bytes = 256} -> !tla.ptr<i8, ub, 256>
    %ub_ptr = tla.recast_ptr %ub_raw : !tla.ptr<i8, ub, 256> -> !tla.ptr<f32, ub, 256>
    %shape = tla.make_shape -> !tla.shape<8,8>
    %stride = tla.make_stride -> !tla.stride<8,1>
    %layout = tla.make_layout %shape, %stride : !tla.shape<8,8>, !tla.stride<8,1> -> !tla.layout<!tla.shape<8,8>, !tla.stride<8,1>, !tla.shape<8,8>, RowMajor>
    %coord = tla.make_coord -> !tla.coord<0,0>
    %ub = tla.make_tensor %ub_ptr, %layout, %coord : !tla.ptr<f32, ub, 256>, !tla.layout<!tla.shape<8,8>, !tla.stride<8,1>, !tla.shape<8,8>, RowMajor>, !tla.coord<0,0> -> !tla.tensor<!tla.layout<!tla.shape<8,8>, !tla.stride<8,1>, !tla.shape<8,8>, RowMajor>, !tla.coord<0,0>, !tla.ptr<f32, ub, 256>>

    %c0 = arith.constant 0 : i64
    %c1 = arith.constant 1 : i64
    %pad = tla.CopyPadParams %c0, %c1, %c0 : i64, i64, i64
    "tla.vector"() ({
      tla.copy %ub, %arg0, %pad : !tla.tensor<!tla.layout<!tla.shape<8,8>, !tla.stride<8,1>, !tla.shape<8,8>, RowMajor>, !tla.coord<0,0>, !tla.ptr<f32, ub, 256>>, !tla.tensor<!tla.layout<!tla.shape<8,7>, !tla.stride<7,1>, !tla.shape<8,7>, RowMajor>, !tla.coord<0,0>, !tla.ptr<f32, gm, 4>>, !tla.copy_pad_params
    }) : () -> ()
    "tla.return"() : () -> ()
  }) {function_type = (!tla.tensor<!tla.layout<!tla.shape<8,7>, !tla.stride<7,1>, !tla.shape<8,7>, RowMajor>, !tla.coord<0,0>, !tla.ptr<f32, gm, 4>>) -> (), sym_name = "copy_gm_ub_pad"} : () -> ()
}

// CHECK: func.func private @copy_gm_RowMajor_to_ub_RowMajor_float
// CHECK-LABEL: func.func @copy_gm_ub_pad
// CHECK: call @copy_gm_RowMajor_to_ub_RowMajor_float
// Trailing ABI: padEnable=1, left=0, right=1, valueBits=0 (after gmCachePolicy).
// CHECK-NOT: "tla.copy"
// CHECK-NOT: "tla.CopyPadParams"
