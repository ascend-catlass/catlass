// RUN: %tla_compile %s -o - | %filecheck %s
//
// HiFloat8 GM→UB VectorLayout two-arg + CopyPadParams (uint8 ExtParams BC).

module {
  "tla.func"() ({
  ^bb0(%arg0: !tla.tensor<!tla.layout<!tla.shape<48>, !tla.stride<1>, !tla.shape<48>, VectorLayout>, !tla.coord<0>, !tla.ptr<!tla.hif8, gm, 1>>):
    %ub_raw = tla.alloc_ptr{size_bytes = 256} -> !tla.ptr<i8, ub, 256>
    %ub_ptr = tla.recast_ptr %ub_raw : !tla.ptr<i8, ub, 256> -> !tla.ptr<!tla.hif8, ub, 256>
    %shape = tla.make_shape -> !tla.shape<56>
    %stride = tla.make_stride -> !tla.stride<1>
    %layout = tla.make_layout %shape, %stride : !tla.shape<56>, !tla.stride<1> -> !tla.layout<!tla.shape<56>, !tla.stride<1>, !tla.shape<56>, VectorLayout>
    %coord = tla.make_coord -> !tla.coord<0>
    %ub = tla.make_tensor %ub_ptr, %layout, %coord : !tla.ptr<!tla.hif8, ub, 256>, !tla.layout<!tla.shape<56>, !tla.stride<1>, !tla.shape<56>, VectorLayout>, !tla.coord<0> -> !tla.tensor<!tla.layout<!tla.shape<56>, !tla.stride<1>, !tla.shape<56>, VectorLayout>, !tla.coord<0>, !tla.ptr<!tla.hif8, ub, 256>>

    %c0 = arith.constant 0 : i64
    %c1 = arith.constant 1 : i64
    %cA5 = arith.constant 165 : i64
    %pad = tla.CopyPadParams %c0, %c1, %cA5 : i64, i64, i64
    "tla.vector"() ({
      tla.copy %ub, %arg0, %pad : !tla.tensor<!tla.layout<!tla.shape<56>, !tla.stride<1>, !tla.shape<56>, VectorLayout>, !tla.coord<0>, !tla.ptr<!tla.hif8, ub, 256>>, !tla.tensor<!tla.layout<!tla.shape<48>, !tla.stride<1>, !tla.shape<48>, VectorLayout>, !tla.coord<0>, !tla.ptr<!tla.hif8, gm, 1>>, !tla.copy_pad_params
    }) : () -> ()
    "tla.return"() : () -> ()
  }) {function_type = (!tla.tensor<!tla.layout<!tla.shape<48>, !tla.stride<1>, !tla.shape<48>, VectorLayout>, !tla.coord<0>, !tla.ptr<!tla.hif8, gm, 1>>) -> (), sym_name = "copy_gm_ub_vl_hif8_pad"} : () -> ()
}

// CHECK: func.func private @copy_gm_VectorLayout_to_ub_VectorLayout_hifloat8_t
// CHECK-SAME: hivm.func_core_type = #hivm.func_core_type<AIV>
// CHECK-LABEL: func.func @copy_gm_ub_vl_hif8_pad
// CHECK: call @copy_gm_VectorLayout_to_ub_VectorLayout_hifloat8_t
