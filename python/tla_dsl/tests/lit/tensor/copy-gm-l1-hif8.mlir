// RUN: %tla_compile %s -o - | %filecheck %s
//
// HiFloat8 GM→L1 RowMajor→zN: typed hifloat8_t BC (parity with GM↔UB RowMajor path).
// IR element is !tla.hif8; memref boundary is i8.

module {
  "tla.func"() ({
  ^bb0(%arg0: !tla.tensor<!tla.layout<!tla.shape<128,128>, !tla.stride<128,1>, !tla.shape<128,128>, RowMajor>, !tla.coord<0,0>, !tla.ptr<!tla.hif8, gm, 1>>):
    %0 = "tla.make_shape"() : () -> !tla.shape<16,64>
    %1 = "tla.make_coord"() : () -> !tla.coord<0,0>
    %2 = "tla.tile_view"(%arg0, %0, %1) : (!tla.tensor<!tla.layout<!tla.shape<128,128>, !tla.stride<128,1>, !tla.shape<128,128>, RowMajor>, !tla.coord<0,0>, !tla.ptr<!tla.hif8, gm, 1>>, !tla.shape<16,64>, !tla.coord<0,0>) -> !tla.tensor<!tla.layout<!tla.shape<16,64>, !tla.stride<128,1>, !tla.shape<16,64>, RowMajor>, !tla.coord<0,0>, !tla.ptr<!tla.hif8, gm, 1>>
    %3 = "tla.alloc_ptr"() {size_bytes = 2048 : i64} : () -> !tla.ptr<i8, l1, 512>
    %4 = "tla.recast_ptr"(%3) : (!tla.ptr<i8, l1, 512>) -> !tla.ptr<!tla.hif8, l1, 512>
    %5 = "tla.make_tensor_like"(%4, %2) {layoutTag = #tla.layout_tag<zN>} : (!tla.ptr<!tla.hif8, l1, 512>, !tla.tensor<!tla.layout<!tla.shape<16,64>, !tla.stride<128,1>, !tla.shape<16,64>, RowMajor>, !tla.coord<0,0>, !tla.ptr<!tla.hif8, gm, 1>>) -> !tla.tensor<!tla.layout<!tla.shape<(16,1),(32,2)>, !tla.stride<(32,512),(1,512)>, !tla.shape<16,64>, zN>, !tla.coord<0,0>, !tla.ptr<!tla.hif8, l1, 512>>
    "tla.cube"() ({
      "tla.copy"(%5, %2) : (!tla.tensor<!tla.layout<!tla.shape<(16,1),(32,2)>, !tla.stride<(32,512),(1,512)>, !tla.shape<16,64>, zN>, !tla.coord<0,0>, !tla.ptr<!tla.hif8, l1, 512>>, !tla.tensor<!tla.layout<!tla.shape<16,64>, !tla.stride<128,1>, !tla.shape<16,64>, RowMajor>, !tla.coord<0,0>, !tla.ptr<!tla.hif8, gm, 1>>) -> ()
    }) : () -> ()
    "tla.return"() : () -> ()
  }) {function_type = (!tla.tensor<!tla.layout<!tla.shape<128,128>, !tla.stride<128,1>, !tla.shape<128,128>, RowMajor>, !tla.coord<0,0>, !tla.ptr<!tla.hif8, gm, 1>>) -> (), sym_name = "copy_gm_l1_hif8"} : () -> ()
}

// CHECK: func.func private @copy_gm_RowMajor_to_l1_zN_hifloat8_t
// CHECK-SAME: hivm.func_core_type = #hivm.func_core_type<AIC>
// CHECK-LABEL: func.func @copy_gm_l1_hif8
// CHECK: call @copy_gm_RowMajor_to_l1_zN_hifloat8_t
// CHECK-NOT: call @copy_gm_RowMajor_to_l1_zN_int8_t
