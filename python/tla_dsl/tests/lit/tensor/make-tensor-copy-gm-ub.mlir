// RUN: %tla_compile %s -o - | %filecheck %s
// RUN: sed 's/f32/ui8/g' %s | %tla_compile - -o - | %filecheck %s --check-prefix=UNSIGNED
// RUN: sed 's/f32/i1/g' %s | %tla_compile - -o - | %filecheck %s --check-prefix=BOOL
// RUN: sed 's/f32/i64/g' %s | %tla_compile - -o - | %filecheck %s --check-prefix=SIGNED64
// RUN: sed 's/f32/ui64/g' %s | %tla_compile - -o - | %filecheck %s --check-prefix=UNSIGNED64
// RUN: sed 's/f32/f64/g' %s | %tla_compile - -o - | %filecheck %s --check-prefix=FLOAT64

module {
  tla.func @make_tensor_copy_kernel(%arg0: !tla.tensor<!tla.layout<!tla.shape<128,128>, !tla.stride<128,1>, !tla.shape<128,128>, RowMajor>, !tla.coord<0,0>, !tla.ptr<f32, gm, 4>>) {
    %0 = tla.make_shape -> !tla.shape<16,16>
    %1 = tla.make_coord -> !tla.coord<0,0>
    %2 = tla.make_coord -> !tla.coord<0,0>
    %3 = tla.tile_view %arg0, %0, %2 : !tla.tensor<!tla.layout<!tla.shape<128,128>, !tla.stride<128,1>, !tla.shape<128,128>, RowMajor>, !tla.coord<0,0>, !tla.ptr<f32, gm, 4>>, !tla.shape<16,16>, !tla.coord<0,0> -> !tla.tensor<!tla.layout<!tla.shape<16,16>, !tla.stride<128,1>, !tla.shape<16,16>, RowMajor>, !tla.coord<0,0>, !tla.ptr<f32, gm, 4>>
    %4 = tla.alloc_ptr{size_bytes = 1024} -> !tla.ptr<i8, ub, 256>
    %5 = tla.recast_ptr %4 : !tla.ptr<i8, ub, 256> -> !tla.ptr<f32, ub, 256>
    %6 = tla.make_shape -> !tla.shape<16,16>
    %7 = tla.make_stride -> !tla.stride<16,1>
    %8 = tla.make_layout %6, %7 : !tla.shape<16,16>, !tla.stride<16,1> -> !tla.layout<!tla.shape<16,16>, !tla.stride<16,1>, !tla.shape<16,16>, RowMajor>
    %9 = tla.make_coord -> !tla.coord<0,0>
    %10 = tla.make_tensor %5, %8, %9 : !tla.ptr<f32, ub, 256>, !tla.layout<!tla.shape<16,16>, !tla.stride<16,1>, !tla.shape<16,16>, RowMajor>, !tla.coord<0,0> -> !tla.tensor<!tla.layout<!tla.shape<16,16>, !tla.stride<16,1>, !tla.shape<16,16>, RowMajor>, !tla.coord<0,0>, !tla.ptr<f32, ub, 256>>
    "tla.vector"() ({
      tla.copy %3, %10 : !tla.tensor<!tla.layout<!tla.shape<16,16>, !tla.stride<128,1>, !tla.shape<16,16>, RowMajor>, !tla.coord<0,0>, !tla.ptr<f32, gm, 4>>, !tla.tensor<!tla.layout<!tla.shape<16,16>, !tla.stride<16,1>, !tla.shape<16,16>, RowMajor>, !tla.coord<0,0>, !tla.ptr<f32, ub, 256>>
      tla.copy %10, %3 : !tla.tensor<!tla.layout<!tla.shape<16,16>, !tla.stride<16,1>, !tla.shape<16,16>, RowMajor>, !tla.coord<0,0>, !tla.ptr<f32, ub, 256>>, !tla.tensor<!tla.layout<!tla.shape<16,16>, !tla.stride<128,1>, !tla.shape<16,16>, RowMajor>, !tla.coord<0,0>, !tla.ptr<f32, gm, 4>>
    }) : () -> ()
    tla.return
  }
}

// CHECK-LABEL: func.func private @copy_ub_RowMajor_to_gm_RowMajor_float
// CHECK-SAME: hivm.func_core_type = #hivm.func_core_type<AIV>
// CHECK-LABEL: func.func @make_tensor_copy_kernel
// CHECK: hivm.hir.pointer_cast{{.*}} : memref<256xf32, #hivm.address_space<ub>>
// CHECK: call @copy_ub_RowMajor_to_gm_RowMajor_float
// CHECK-NOT: tla.make_tensor
// CHECK-NOT: memref.reinterpret_cast
// An unsigned tile stages across GM<->UB under its own symbol. The move is the
// signed one at the same width -- a copy does not interpret the element type
// beyond stride and layout arithmetic -- but the route resolves by element type,
// so the symbol has to exist or an unsigned tile cannot reach UB at all.
// UNSIGNED: func.func private @copy_ub_RowMajor_to_gm_RowMajor_uint8_t
// UNSIGNED: call @copy_ub_RowMajor_to_gm_RowMajor_uint8_t

// Bool / i1 stages as AscendC b8 under the ``bool`` BC symbol
// (copyRuntimeElemSuffix → "bool"; Vector/dma.cpp REGISTER_*(..., bool)).
// BOOL-DAG: call @copy_ub_RowMajor_to_gm_RowMajor_bool
// BOOL-DAG: call @copy_gm_RowMajor_to_ub_RowMajor_bool

// SIGNED64-DAG: call @copy_ub_RowMajor_to_gm_RowMajor_int64_t
// SIGNED64-DAG: call @copy_gm_RowMajor_to_ub_RowMajor_int64_t
// UNSIGNED64-DAG: call @copy_ub_RowMajor_to_gm_RowMajor_uint64_t
// UNSIGNED64-DAG: call @copy_gm_RowMajor_to_ub_RowMajor_uint64_t
// FLOAT64-DAG: call @copy_ub_RowMajor_to_gm_RowMajor_double
// FLOAT64-DAG: call @copy_gm_RowMajor_to_ub_RowMajor_double
