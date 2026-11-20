// RUN: %tla_compile %s -o - | %filecheck %s
//
// HiFloat8 GM↔UB RowMajor: typed hifloat8_t BC symbols (tile uint8 DataCopyPad).
// IR element is !tla.hif8; memref boundary is i8 (isTlaCustomElementType).

module {
  tla.func @copy_gm_ub_hif8(
      %arg0: !tla.tensor<!tla.layout<!tla.shape<128,128>, !tla.stride<128,1>, !tla.shape<128,128>, RowMajor>, !tla.coord<0,0>, !tla.ptr<!tla.hif8, gm, 1>>) {
    %0 = tla.make_shape -> !tla.shape<16,16>
    %1 = tla.make_coord -> !tla.coord<0,0>
    %2 = tla.tile_view %arg0, %0, %1 : !tla.tensor<!tla.layout<!tla.shape<128,128>, !tla.stride<128,1>, !tla.shape<128,128>, RowMajor>, !tla.coord<0,0>, !tla.ptr<!tla.hif8, gm, 1>>, !tla.shape<16,16>, !tla.coord<0,0> -> !tla.tensor<!tla.layout<!tla.shape<16,16>, !tla.stride<128,1>, !tla.shape<16,16>, RowMajor>, !tla.coord<0,0>, !tla.ptr<!tla.hif8, gm, 1>>
    %3 = tla.alloc_ptr{size_bytes = 1024} -> !tla.ptr<i8, ub, 256>
    %4 = tla.recast_ptr %3 : !tla.ptr<i8, ub, 256> -> !tla.ptr<!tla.hif8, ub, 256>
    %5 = tla.make_shape -> !tla.shape<16,16>
    %6 = tla.make_stride -> !tla.stride<16,1>
    %7 = tla.make_layout %5, %6 : !tla.shape<16,16>, !tla.stride<16,1> -> !tla.layout<!tla.shape<16,16>, !tla.stride<16,1>, !tla.shape<16,16>, RowMajor>
    %8 = tla.make_coord -> !tla.coord<0,0>
    %9 = tla.make_tensor %4, %7, %8 : !tla.ptr<!tla.hif8, ub, 256>, !tla.layout<!tla.shape<16,16>, !tla.stride<16,1>, !tla.shape<16,16>, RowMajor>, !tla.coord<0,0> -> !tla.tensor<!tla.layout<!tla.shape<16,16>, !tla.stride<16,1>, !tla.shape<16,16>, RowMajor>, !tla.coord<0,0>, !tla.ptr<!tla.hif8, ub, 256>>
    "tla.vector"() ({
      tla.copy %2, %9 : !tla.tensor<!tla.layout<!tla.shape<16,16>, !tla.stride<128,1>, !tla.shape<16,16>, RowMajor>, !tla.coord<0,0>, !tla.ptr<!tla.hif8, gm, 1>>, !tla.tensor<!tla.layout<!tla.shape<16,16>, !tla.stride<16,1>, !tla.shape<16,16>, RowMajor>, !tla.coord<0,0>, !tla.ptr<!tla.hif8, ub, 256>>
      tla.copy %9, %2 : !tla.tensor<!tla.layout<!tla.shape<16,16>, !tla.stride<16,1>, !tla.shape<16,16>, RowMajor>, !tla.coord<0,0>, !tla.ptr<!tla.hif8, ub, 256>>, !tla.tensor<!tla.layout<!tla.shape<16,16>, !tla.stride<128,1>, !tla.shape<16,16>, RowMajor>, !tla.coord<0,0>, !tla.ptr<!tla.hif8, gm, 1>>
    }) : () -> ()
    tla.return
  }
}

// Decl order follows BC emission (ub→gm then gm→ub).
// CHECK: func.func private @copy_ub_RowMajor_to_gm_RowMajor_hifloat8_t
// CHECK-SAME: hivm.func_core_type = #hivm.func_core_type<AIV>
// CHECK: func.func private @copy_gm_RowMajor_to_ub_RowMajor_hifloat8_t
// CHECK-SAME: hivm.func_core_type = #hivm.func_core_type<AIV>
// CHECK-LABEL: func.func @copy_gm_ub_hif8
// CHECK-DAG: call @copy_gm_RowMajor_to_ub_RowMajor_hifloat8_t
// CHECK-DAG: call @copy_ub_RowMajor_to_gm_RowMajor_hifloat8_t
// CHECK-NOT: call @copy_gm_RowMajor_to_ub_RowMajor_int8_t
// CHECK-NOT: call @copy_ub_RowMajor_to_gm_RowMajor_int8_t
