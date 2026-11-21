// Bool GM→L1 fractal routes (typed REGISTER_GM_TO_L1(..., bool)): RowMajor→zN / ColumnMajor→nZ.
// Must resolve the ``bool`` symbol with llvm.emit_c_interface (AIC), not silent i8 fallback.

// RUN: split-file %s %t
// RUN: %tla_compile %t/zn.mlir -o - 2>&1 | %filecheck %s --check-prefix=ZN
// RUN: %tla_compile %t/nz.mlir -o - 2>&1 | %filecheck %s --check-prefix=NZ

//--- zn.mlir

module {
  tla.func @copy_gm_rowmajor_to_l1_zn_bool(
      %arg0: !tla.tensor<!tla.layout<!tla.shape<32,32>, !tla.stride<32,1>, !tla.shape<32,32>, RowMajor>, !tla.coord<0,0>, !tla.ptr<i1, gm, 1>>) {
    %0 = tla.alloc_ptr{size_bytes = 1024} -> !tla.ptr<i1, l1, 512>
    %1 = tla.make_tensor_like %0 like %arg0 layoutTag(<zN>) : !tla.ptr<i1, l1, 512>, !tla.tensor<!tla.layout<!tla.shape<32,32>, !tla.stride<32,1>, !tla.shape<32,32>, RowMajor>, !tla.coord<0,0>, !tla.ptr<i1, gm, 1>> -> !tla.tensor<!tla.layout<!tla.shape<(16,2),(32,1)>, !tla.stride<(32,512),(1,1024)>, !tla.shape<32,32>, zN>, !tla.coord<0,0>, !tla.ptr<i1, l1, 512>>
    tla.cube {
      tla.copy %1, %arg0 : !tla.tensor<!tla.layout<!tla.shape<(16,2),(32,1)>, !tla.stride<(32,512),(1,1024)>, !tla.shape<32,32>, zN>, !tla.coord<0,0>, !tla.ptr<i1, l1, 512>>, !tla.tensor<!tla.layout<!tla.shape<32,32>, !tla.stride<32,1>, !tla.shape<32,32>, RowMajor>, !tla.coord<0,0>, !tla.ptr<i1, gm, 1>>
    }
    tla.return
  }
}

// ZN-DAG: func.func private @copy_gm_RowMajor_to_l1_zN_bool
// ZN-DAG: attributes {{{.*}}llvm.emit_c_interface{{.*}}}
// ZN-DAG: hivm.func_core_type = #hivm.func_core_type<AIC>
// ZN: call @copy_gm_RowMajor_to_l1_zN_bool
// ZN-NOT: copy_gm_RowMajor_to_l1_zN_int8_t

//--- nz.mlir

module {
  tla.func @copy_gm_columnmajor_to_l1_nz_bool(
      %arg0: !tla.tensor<!tla.layout<!tla.shape<32,32>, !tla.stride<1,32>, !tla.shape<32,32>, ColumnMajor>, !tla.coord<0,0>, !tla.ptr<i1, gm, 1>>) {
    %0 = tla.alloc_ptr{size_bytes = 1024} -> !tla.ptr<i1, l1, 512>
    %1 = tla.make_tensor_like %0 like %arg0 layoutTag(<nZ>) : !tla.ptr<i1, l1, 512>, !tla.tensor<!tla.layout<!tla.shape<32,32>, !tla.stride<1,32>, !tla.shape<32,32>, ColumnMajor>, !tla.coord<0,0>, !tla.ptr<i1, gm, 1>> -> !tla.tensor<!tla.layout<!tla.shape<(32,1),(16,2)>, !tla.stride<(1,1024),(32,512)>, !tla.shape<32,32>, nZ>, !tla.coord<0,0>, !tla.ptr<i1, l1, 512>>
    tla.cube {
      tla.copy %1, %arg0 : !tla.tensor<!tla.layout<!tla.shape<(32,1),(16,2)>, !tla.stride<(1,1024),(32,512)>, !tla.shape<32,32>, nZ>, !tla.coord<0,0>, !tla.ptr<i1, l1, 512>>, !tla.tensor<!tla.layout<!tla.shape<32,32>, !tla.stride<1,32>, !tla.shape<32,32>, ColumnMajor>, !tla.coord<0,0>, !tla.ptr<i1, gm, 1>>
    }
    tla.return
  }
}

// NZ-DAG: func.func private @copy_gm_ColumnMajor_to_l1_nZ_bool
// NZ-DAG: attributes {{{.*}}llvm.emit_c_interface{{.*}}}
// NZ-DAG: hivm.func_core_type = #hivm.func_core_type<AIC>
// NZ: call @copy_gm_ColumnMajor_to_l1_nZ_bool
// NZ-NOT: copy_gm_ColumnMajor_to_l1_nZ_int8_t
