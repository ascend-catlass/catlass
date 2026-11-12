// A layout-dynamic scale tile cannot be checked at compile time: the number of
// dequant elements it holds is only known when the kernel runs. The front end
// refuses a static length mismatch, and passes a dynamic length through;
// matching the destination tile's channels is then the caller's contract. This
// pins the pass-through lowering of that contract.
//
// RUN: %tla_compile %s -o - | %filecheck %s

module attributes {tla.module_exec_units = "cube"} {
  "tla.func"() ({
  ^bb0(%arg0: !tla.tensor<!tla.layout<!tla.shape<32,32>, !tla.stride<32,1>, !tla.shape<32,32>, RowMajor>, !tla.coord<0,0>, !tla.ptr<f16, gm, 4>>, %n: index):
    %0 = "tla.make_shape"() : () -> !tla.shape<32,32>
    %1 = "tla.make_coord"() : () -> !tla.coord<0,0>
    %2 = "tla.make_coord"() : () -> !tla.coord<0,0>
    %3 = "tla.tile_view"(%arg0, %0, %2) : (!tla.tensor<!tla.layout<!tla.shape<32,32>, !tla.stride<32,1>, !tla.shape<32,32>, RowMajor>, !tla.coord<0,0>, !tla.ptr<f16, gm, 4>>, !tla.shape<32,32>, !tla.coord<0,0>) -> !tla.tensor<!tla.layout<!tla.shape<32,32>, !tla.stride<32,1>, !tla.shape<32,32>, RowMajor>, !tla.coord<0,0>, !tla.ptr<f16, gm, 4>>

    %4 = "tla.alloc_ptr"() {size_bytes = 4096 : i64} : () -> !tla.ptr<i8, l0c, 1024>
    %5 = "tla.recast_ptr"(%4) : (!tla.ptr<i8, l0c, 1024>) -> !tla.ptr<f32, l0c, 1024>
    %6 = "tla.make_tensor_like"(%5, %3) {layoutTag = #tla.layout_tag<L0Clayout>} : (!tla.ptr<f32, l0c, 1024>, !tla.tensor<!tla.layout<!tla.shape<32,32>, !tla.stride<32,1>, !tla.shape<32,32>, RowMajor>, !tla.coord<0,0>, !tla.ptr<f16, gm, 4>>) -> !tla.tensor<!tla.layout<!tla.shape<(16,2),(16,2)>, !tla.stride<(16,256),(1,512)>, !tla.shape<32,32>, L0Clayout>, !tla.coord<0,0>, !tla.ptr<f32, l0c, 1024>>

    // The dequant run's length is the runtime `%n`, not a literal.
    %7 = "tla.alloc_ptr"() {size_bytes = 4096 : i64} : () -> !tla.ptr<i8, l1, 32>
    %8 = "tla.recast_ptr"(%7) : (!tla.ptr<i8, l1, 32>) -> !tla.ptr<ui64, l1, 32>
    %9 = "tla.make_shape"(%n) : (index) -> !tla.shape<?>
    %10 = "tla.make_stride"() : () -> !tla.stride<1>
    %11 = "tla.make_layout"(%9, %10) : (!tla.shape<?>, !tla.stride<1>) -> !tla.layout<!tla.shape<?>, !tla.stride<1>, !tla.shape<?>, VectorLayout>
    %12 = "tla.make_coord"() : () -> !tla.coord<0>
    %quant = "tla.make_tensor"(%8, %11, %12) : (!tla.ptr<ui64, l1, 32>, !tla.layout<!tla.shape<?>, !tla.stride<1>, !tla.shape<?>, VectorLayout>, !tla.coord<0>) -> !tla.tensor<!tla.layout<!tla.shape<?>, !tla.stride<1>, !tla.shape<?>, VectorLayout>, !tla.coord<0>, !tla.ptr<ui64, l1, 32>>

    %13 = "tla.CopyL0C2DstParams"(%quant) <{unit_flag = 0 : i64, relu_enable = false, quant_mode = #tla.quant_mode<PER_CHANNEL>, l0c2ub_mode = #tla.l0c2ub_mode<NO_SPLIT_VEC_0>, operandSegmentSizes = array<i32: 0, 0, 1>}> : (!tla.tensor<!tla.layout<!tla.shape<?>, !tla.stride<1>, !tla.shape<?>, VectorLayout>, !tla.coord<0>, !tla.ptr<ui64, l1, 32>>) -> !tla.copy_l0c2dst_params

    "tla.cube"() ({
      "tla.copy"(%3, %6, %13) : (!tla.tensor<!tla.layout<!tla.shape<32,32>, !tla.stride<32,1>, !tla.shape<32,32>, RowMajor>, !tla.coord<0,0>, !tla.ptr<f16, gm, 4>>, !tla.tensor<!tla.layout<!tla.shape<(16,2),(16,2)>, !tla.stride<(16,256),(1,512)>, !tla.shape<32,32>, L0Clayout>, !tla.coord<0,0>, !tla.ptr<f32, l0c, 1024>>, !tla.copy_l0c2dst_params) -> ()
    }) : () -> ()
    "tla.return"() : () -> ()
  }) {tla.exec_units = "cube", function_type = (!tla.tensor<!tla.layout<!tla.shape<32,32>, !tla.stride<32,1>, !tla.shape<32,32>, RowMajor>, !tla.coord<0,0>, !tla.ptr<f16, gm, 4>>, index) -> (), sym_name = "copy_l0c2gm_perchannel_dynamic_scale"} : () -> ()
}

// The dynamic length is passed through: no length diagnostic, and the copy still
// lowers to the per-channel runtime symbol.
// CHECK: func.func private @copy_l0c_to_gm_RowMajor_perchannel_float_half
// CHECK-LABEL: func.func @copy_l0c2gm_perchannel_dynamic_scale
// CHECK: call @copy_l0c_to_gm_RowMajor_perchannel_float_half
// CHECK-NOT: elements but the destination tile spans
// CHECK-NOT: "tla.copy"
