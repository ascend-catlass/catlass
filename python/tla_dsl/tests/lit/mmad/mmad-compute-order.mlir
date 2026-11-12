// Verify the compute_order attribute on tla.mmad lowers to SPR.CTRL[51].

// Owned by tla-insert-implicit-ctrl (InitSoc leaves CTRL[51]=0 / M_FIRST):
// N_FIRST  -> hivm.hir.set_ctrl true  at ctrl[51], restored after.
// default  -> no insert (already matches InitSoc); epilogue still clears.
// dual N_FIRST -> one set_ctrl true at ctrl[51] covers both mmads.
// mixed (N_FIRST + M_FIRST) -> per-mmad set_ctrl at ctrl[51].
//
// RUN: %tla_compile %s -o - | %filecheck %s --check-prefix=NFIRST
// RUN: %tla_compile %s -o - | %filecheck %s --check-prefix=DEFAULT
// RUN: %tla_compile %s -o - | %filecheck %s --check-prefix=DUAL
// RUN: %tla_compile %s -o - | %filecheck %s --check-prefix=MIXED

module {
  tla.func @mmad_nfirst() {
    %0 = tla.alloc_ptr{size_bytes = 4096} -> !tla.ptr<f32, l1, 512>
    %1 = tla.make_shape -> !tla.shape<32,32>
    %2 = tla.make_stride -> !tla.stride<32,1>
    %3 = tla.make_layout %1, %2 : !tla.shape<32,32>, !tla.stride<32,1> -> !tla.layout<!tla.shape<32,32>, !tla.stride<32,1>, !tla.shape<32,32>, RowMajor>
    %4 = tla.make_coord -> !tla.coord<0,0>
    %5 = tla.make_tensor %0, %3, %4 : !tla.ptr<f32, l1, 512>, !tla.layout<!tla.shape<32,32>, !tla.stride<32,1>, !tla.shape<32,32>, RowMajor>, !tla.coord<0,0> -> !tla.tensor<!tla.layout<!tla.shape<32,32>, !tla.stride<32,1>, !tla.shape<32,32>, RowMajor>, !tla.coord<0,0>, !tla.ptr<f32, l1, 512>>
    %6 = tla.alloc_ptr{size_bytes = 4096} -> !tla.ptr<f32, l0a, 512>
    %7 = tla.make_tensor_like %6 like %5 layoutTag(#tla.layout_tag<zN>) : !tla.ptr<f32, l0a, 512>, !tla.tensor<!tla.layout<!tla.shape<32,32>, !tla.stride<32,1>, !tla.shape<32,32>, RowMajor>, !tla.coord<0,0>, !tla.ptr<f32, l1, 512>> -> !tla.tensor<!tla.layout<!tla.shape<(16,2),(8,4)>, !tla.stride<(8,128),(1,256)>, !tla.shape<32,32>, zN>, !tla.coord<0,0>, !tla.ptr<f32, l0a, 512>>
    %8 = tla.alloc_ptr{size_bytes = 4096} -> !tla.ptr<f32, l0b, 512>
    %9 = tla.make_tensor_like %8 like %5 layoutTag(#tla.layout_tag<nZ>) : !tla.ptr<f32, l0b, 512>, !tla.tensor<!tla.layout<!tla.shape<32,32>, !tla.stride<32,1>, !tla.shape<32,32>, RowMajor>, !tla.coord<0,0>, !tla.ptr<f32, l1, 512>> -> !tla.tensor<!tla.layout<!tla.shape<(8,4),(16,2)>, !tla.stride<(1,256),(8,128)>, !tla.shape<32,32>, nZ>, !tla.coord<0,0>, !tla.ptr<f32, l0b, 512>>
    %10 = tla.alloc_ptr{size_bytes = 4096} -> !tla.ptr<f32, l0c, 1024>
    %11 = tla.make_tensor_like %10 like %5 layoutTag(#tla.layout_tag<L0Clayout>) : !tla.ptr<f32, l0c, 1024>, !tla.tensor<!tla.layout<!tla.shape<32,32>, !tla.stride<32,1>, !tla.shape<32,32>, RowMajor>, !tla.coord<0,0>, !tla.ptr<f32, l1, 512>> -> !tla.tensor<!tla.layout<!tla.shape<(16,2),(16,2)>, !tla.stride<(16,256),(1,512)>, !tla.shape<32,32>, L0Clayout>, !tla.coord<0,0>, !tla.ptr<f32, l0c, 1024>>
    tla.cube {
      %true = arith.constant true
      %c3_i64 = arith.constant 3 : i64
      tla.mmad %11, %7, %9, %true, %c3_i64 {compute_order = #tla.compute_order<N_FIRST>, hf32_mode = #tla.hf32_mode<HF32_DISABLE>} : !tla.tensor<!tla.layout<!tla.shape<(16,2),(16,2)>, !tla.stride<(16,256),(1,512)>, !tla.shape<32,32>, L0Clayout>, !tla.coord<0,0>, !tla.ptr<f32, l0c, 1024>>, !tla.tensor<!tla.layout<!tla.shape<(16,2),(8,4)>, !tla.stride<(8,128),(1,256)>, !tla.shape<32,32>, zN>, !tla.coord<0,0>, !tla.ptr<f32, l0a, 512>>, !tla.tensor<!tla.layout<!tla.shape<(8,4),(16,2)>, !tla.stride<(1,256),(8,128)>, !tla.shape<32,32>, nZ>, !tla.coord<0,0>, !tla.ptr<f32, l0b, 512>>, i1, i64
    }
    tla.return
  }
  tla.func @mmad_default() {
    %0 = tla.alloc_ptr{size_bytes = 4096} -> !tla.ptr<f32, l1, 512>
    %1 = tla.make_shape -> !tla.shape<32,32>
    %2 = tla.make_stride -> !tla.stride<32,1>
    %3 = tla.make_layout %1, %2 : !tla.shape<32,32>, !tla.stride<32,1> -> !tla.layout<!tla.shape<32,32>, !tla.stride<32,1>, !tla.shape<32,32>, RowMajor>
    %4 = tla.make_coord -> !tla.coord<0,0>
    %5 = tla.make_tensor %0, %3, %4 : !tla.ptr<f32, l1, 512>, !tla.layout<!tla.shape<32,32>, !tla.stride<32,1>, !tla.shape<32,32>, RowMajor>, !tla.coord<0,0> -> !tla.tensor<!tla.layout<!tla.shape<32,32>, !tla.stride<32,1>, !tla.shape<32,32>, RowMajor>, !tla.coord<0,0>, !tla.ptr<f32, l1, 512>>
    %6 = tla.alloc_ptr{size_bytes = 4096} -> !tla.ptr<f32, l0a, 512>
    %7 = tla.make_tensor_like %6 like %5 layoutTag(#tla.layout_tag<zN>) : !tla.ptr<f32, l0a, 512>, !tla.tensor<!tla.layout<!tla.shape<32,32>, !tla.stride<32,1>, !tla.shape<32,32>, RowMajor>, !tla.coord<0,0>, !tla.ptr<f32, l1, 512>> -> !tla.tensor<!tla.layout<!tla.shape<(16,2),(8,4)>, !tla.stride<(8,128),(1,256)>, !tla.shape<32,32>, zN>, !tla.coord<0,0>, !tla.ptr<f32, l0a, 512>>
    %8 = tla.alloc_ptr{size_bytes = 4096} -> !tla.ptr<f32, l0b, 512>
    %9 = tla.make_tensor_like %8 like %5 layoutTag(#tla.layout_tag<nZ>) : !tla.ptr<f32, l0b, 512>, !tla.tensor<!tla.layout<!tla.shape<32,32>, !tla.stride<32,1>, !tla.shape<32,32>, RowMajor>, !tla.coord<0,0>, !tla.ptr<f32, l1, 512>> -> !tla.tensor<!tla.layout<!tla.shape<(8,4),(16,2)>, !tla.stride<(1,256),(8,128)>, !tla.shape<32,32>, nZ>, !tla.coord<0,0>, !tla.ptr<f32, l0b, 512>>
    %10 = tla.alloc_ptr{size_bytes = 4096} -> !tla.ptr<f32, l0c, 1024>
    %11 = tla.make_tensor_like %10 like %5 layoutTag(#tla.layout_tag<L0Clayout>) : !tla.ptr<f32, l0c, 1024>, !tla.tensor<!tla.layout<!tla.shape<32,32>, !tla.stride<32,1>, !tla.shape<32,32>, RowMajor>, !tla.coord<0,0>, !tla.ptr<f32, l1, 512>> -> !tla.tensor<!tla.layout<!tla.shape<(16,2),(16,2)>, !tla.stride<(16,256),(1,512)>, !tla.shape<32,32>, L0Clayout>, !tla.coord<0,0>, !tla.ptr<f32, l0c, 1024>>
    tla.cube {
      %true = arith.constant true
      %c3_i64 = arith.constant 3 : i64
      tla.mmad %11, %7, %9, %true, %c3_i64 {compute_order = #tla.compute_order<M_FIRST>, hf32_mode = #tla.hf32_mode<HF32_DISABLE>} : !tla.tensor<!tla.layout<!tla.shape<(16,2),(16,2)>, !tla.stride<(16,256),(1,512)>, !tla.shape<32,32>, L0Clayout>, !tla.coord<0,0>, !tla.ptr<f32, l0c, 1024>>, !tla.tensor<!tla.layout<!tla.shape<(16,2),(8,4)>, !tla.stride<(8,128),(1,256)>, !tla.shape<32,32>, zN>, !tla.coord<0,0>, !tla.ptr<f32, l0a, 512>>, !tla.tensor<!tla.layout<!tla.shape<(8,4),(16,2)>, !tla.stride<(1,256),(8,128)>, !tla.shape<32,32>, nZ>, !tla.coord<0,0>, !tla.ptr<f32, l0b, 512>>, i1, i64
    }
    tla.return
  }
  tla.func @mmad_dual_nfirst() {
    %0 = tla.alloc_ptr{size_bytes = 4096} -> !tla.ptr<f32, l1, 512>
    %1 = tla.make_shape -> !tla.shape<32,32>
    %2 = tla.make_stride -> !tla.stride<32,1>
    %3 = tla.make_layout %1, %2 : !tla.shape<32,32>, !tla.stride<32,1> -> !tla.layout<!tla.shape<32,32>, !tla.stride<32,1>, !tla.shape<32,32>, RowMajor>
    %4 = tla.make_coord -> !tla.coord<0,0>
    %5 = tla.make_tensor %0, %3, %4 : !tla.ptr<f32, l1, 512>, !tla.layout<!tla.shape<32,32>, !tla.stride<32,1>, !tla.shape<32,32>, RowMajor>, !tla.coord<0,0> -> !tla.tensor<!tla.layout<!tla.shape<32,32>, !tla.stride<32,1>, !tla.shape<32,32>, RowMajor>, !tla.coord<0,0>, !tla.ptr<f32, l1, 512>>
    %6 = tla.alloc_ptr{size_bytes = 4096} -> !tla.ptr<f32, l0a, 512>
    %7 = tla.make_tensor_like %6 like %5 layoutTag(#tla.layout_tag<zN>) : !tla.ptr<f32, l0a, 512>, !tla.tensor<!tla.layout<!tla.shape<32,32>, !tla.stride<32,1>, !tla.shape<32,32>, RowMajor>, !tla.coord<0,0>, !tla.ptr<f32, l1, 512>> -> !tla.tensor<!tla.layout<!tla.shape<(16,2),(8,4)>, !tla.stride<(8,128),(1,256)>, !tla.shape<32,32>, zN>, !tla.coord<0,0>, !tla.ptr<f32, l0a, 512>>
    %8 = tla.alloc_ptr{size_bytes = 4096} -> !tla.ptr<f32, l0b, 512>
    %9 = tla.make_tensor_like %8 like %5 layoutTag(#tla.layout_tag<nZ>) : !tla.ptr<f32, l0b, 512>, !tla.tensor<!tla.layout<!tla.shape<32,32>, !tla.stride<32,1>, !tla.shape<32,32>, RowMajor>, !tla.coord<0,0>, !tla.ptr<f32, l1, 512>> -> !tla.tensor<!tla.layout<!tla.shape<(8,4),(16,2)>, !tla.stride<(1,256),(8,128)>, !tla.shape<32,32>, nZ>, !tla.coord<0,0>, !tla.ptr<f32, l0b, 512>>
    %10 = tla.alloc_ptr{size_bytes = 4096} -> !tla.ptr<f32, l0c, 1024>
    %11 = tla.make_tensor_like %10 like %5 layoutTag(#tla.layout_tag<L0Clayout>) : !tla.ptr<f32, l0c, 1024>, !tla.tensor<!tla.layout<!tla.shape<32,32>, !tla.stride<32,1>, !tla.shape<32,32>, RowMajor>, !tla.coord<0,0>, !tla.ptr<f32, l1, 512>> -> !tla.tensor<!tla.layout<!tla.shape<(16,2),(16,2)>, !tla.stride<(16,256),(1,512)>, !tla.shape<32,32>, L0Clayout>, !tla.coord<0,0>, !tla.ptr<f32, l0c, 1024>>
    tla.cube {
      %true = arith.constant true
      %false = arith.constant false
      %c3_i64 = arith.constant 3 : i64
      tla.mmad %11, %7, %9, %true, %c3_i64 {compute_order = #tla.compute_order<N_FIRST>, hf32_mode = #tla.hf32_mode<HF32_DISABLE>} : !tla.tensor<!tla.layout<!tla.shape<(16,2),(16,2)>, !tla.stride<(16,256),(1,512)>, !tla.shape<32,32>, L0Clayout>, !tla.coord<0,0>, !tla.ptr<f32, l0c, 1024>>, !tla.tensor<!tla.layout<!tla.shape<(16,2),(8,4)>, !tla.stride<(8,128),(1,256)>, !tla.shape<32,32>, zN>, !tla.coord<0,0>, !tla.ptr<f32, l0a, 512>>, !tla.tensor<!tla.layout<!tla.shape<(8,4),(16,2)>, !tla.stride<(1,256),(8,128)>, !tla.shape<32,32>, nZ>, !tla.coord<0,0>, !tla.ptr<f32, l0b, 512>>, i1, i64
      tla.mmad %11, %7, %9, %false, %c3_i64 {compute_order = #tla.compute_order<N_FIRST>, hf32_mode = #tla.hf32_mode<HF32_DISABLE>} : !tla.tensor<!tla.layout<!tla.shape<(16,2),(16,2)>, !tla.stride<(16,256),(1,512)>, !tla.shape<32,32>, L0Clayout>, !tla.coord<0,0>, !tla.ptr<f32, l0c, 1024>>, !tla.tensor<!tla.layout<!tla.shape<(16,2),(8,4)>, !tla.stride<(8,128),(1,256)>, !tla.shape<32,32>, zN>, !tla.coord<0,0>, !tla.ptr<f32, l0a, 512>>, !tla.tensor<!tla.layout<!tla.shape<(8,4),(16,2)>, !tla.stride<(1,256),(8,128)>, !tla.shape<32,32>, nZ>, !tla.coord<0,0>, !tla.ptr<f32, l0b, 512>>, i1, i64
    }
    tla.return
  }
  tla.func @mmad_mixed_compute_order() {
    %0 = tla.alloc_ptr{size_bytes = 4096} -> !tla.ptr<f32, l1, 512>
    %1 = tla.make_shape -> !tla.shape<32,32>
    %2 = tla.make_stride -> !tla.stride<32,1>
    %3 = tla.make_layout %1, %2 : !tla.shape<32,32>, !tla.stride<32,1> -> !tla.layout<!tla.shape<32,32>, !tla.stride<32,1>, !tla.shape<32,32>, RowMajor>
    %4 = tla.make_coord -> !tla.coord<0,0>
    %5 = tla.make_tensor %0, %3, %4 : !tla.ptr<f32, l1, 512>, !tla.layout<!tla.shape<32,32>, !tla.stride<32,1>, !tla.shape<32,32>, RowMajor>, !tla.coord<0,0> -> !tla.tensor<!tla.layout<!tla.shape<32,32>, !tla.stride<32,1>, !tla.shape<32,32>, RowMajor>, !tla.coord<0,0>, !tla.ptr<f32, l1, 512>>
    %6 = tla.alloc_ptr{size_bytes = 4096} -> !tla.ptr<f32, l0a, 512>
    %7 = tla.make_tensor_like %6 like %5 layoutTag(#tla.layout_tag<zN>) : !tla.ptr<f32, l0a, 512>, !tla.tensor<!tla.layout<!tla.shape<32,32>, !tla.stride<32,1>, !tla.shape<32,32>, RowMajor>, !tla.coord<0,0>, !tla.ptr<f32, l1, 512>> -> !tla.tensor<!tla.layout<!tla.shape<(16,2),(8,4)>, !tla.stride<(8,128),(1,256)>, !tla.shape<32,32>, zN>, !tla.coord<0,0>, !tla.ptr<f32, l0a, 512>>
    %8 = tla.alloc_ptr{size_bytes = 4096} -> !tla.ptr<f32, l0b, 512>
    %9 = tla.make_tensor_like %8 like %5 layoutTag(#tla.layout_tag<nZ>) : !tla.ptr<f32, l0b, 512>, !tla.tensor<!tla.layout<!tla.shape<32,32>, !tla.stride<32,1>, !tla.shape<32,32>, RowMajor>, !tla.coord<0,0>, !tla.ptr<f32, l1, 512>> -> !tla.tensor<!tla.layout<!tla.shape<(8,4),(16,2)>, !tla.stride<(1,256),(8,128)>, !tla.shape<32,32>, nZ>, !tla.coord<0,0>, !tla.ptr<f32, l0b, 512>>
    %10 = tla.alloc_ptr{size_bytes = 4096} -> !tla.ptr<f32, l0c, 1024>
    %11 = tla.make_tensor_like %10 like %5 layoutTag(#tla.layout_tag<L0Clayout>) : !tla.ptr<f32, l0c, 1024>, !tla.tensor<!tla.layout<!tla.shape<32,32>, !tla.stride<32,1>, !tla.shape<32,32>, RowMajor>, !tla.coord<0,0>, !tla.ptr<f32, l1, 512>> -> !tla.tensor<!tla.layout<!tla.shape<(16,2),(16,2)>, !tla.stride<(16,256),(1,512)>, !tla.shape<32,32>, L0Clayout>, !tla.coord<0,0>, !tla.ptr<f32, l0c, 1024>>
    tla.cube {
      %true = arith.constant true
      %false = arith.constant false
      %c3_i64 = arith.constant 3 : i64
      tla.mmad %11, %7, %9, %true, %c3_i64 {compute_order = #tla.compute_order<N_FIRST>, hf32_mode = #tla.hf32_mode<HF32_DISABLE>} : !tla.tensor<!tla.layout<!tla.shape<(16,2),(16,2)>, !tla.stride<(16,256),(1,512)>, !tla.shape<32,32>, L0Clayout>, !tla.coord<0,0>, !tla.ptr<f32, l0c, 1024>>, !tla.tensor<!tla.layout<!tla.shape<(16,2),(8,4)>, !tla.stride<(8,128),(1,256)>, !tla.shape<32,32>, zN>, !tla.coord<0,0>, !tla.ptr<f32, l0a, 512>>, !tla.tensor<!tla.layout<!tla.shape<(8,4),(16,2)>, !tla.stride<(1,256),(8,128)>, !tla.shape<32,32>, nZ>, !tla.coord<0,0>, !tla.ptr<f32, l0b, 512>>, i1, i64
      tla.mmad %11, %7, %9, %false, %c3_i64 {compute_order = #tla.compute_order<M_FIRST>, hf32_mode = #tla.hf32_mode<HF32_DISABLE>} : !tla.tensor<!tla.layout<!tla.shape<(16,2),(16,2)>, !tla.stride<(16,256),(1,512)>, !tla.shape<32,32>, L0Clayout>, !tla.coord<0,0>, !tla.ptr<f32, l0c, 1024>>, !tla.tensor<!tla.layout<!tla.shape<(16,2),(8,4)>, !tla.stride<(8,128),(1,256)>, !tla.shape<32,32>, zN>, !tla.coord<0,0>, !tla.ptr<f32, l0a, 512>>, !tla.tensor<!tla.layout<!tla.shape<(8,4),(16,2)>, !tla.stride<(1,256),(8,128)>, !tla.shape<32,32>, nZ>, !tla.coord<0,0>, !tla.ptr<f32, l0b, 512>>, i1, i64
    }
    tla.return
  }
}

// NFIRST-LABEL: func.func @mmad_nfirst
// NFIRST: hivm.hir.set_ctrl true at ctrl[51]
// NFIRST: call @mmad_float_float_float
// NFIRST: hivm.hir.set_ctrl false at ctrl[51]
// NFIRST-NOT: tla.mmad

// DEFAULT-LABEL: func.func @mmad_default
// DEFAULT-NOT: hivm.hir.set_ctrl true at ctrl[51]
// DEFAULT: call @mmad_float_float_float
// DEFAULT: hivm.hir.set_ctrl false at ctrl[51]
// DEFAULT-NOT: tla.mmad

// DUAL-LABEL: func.func @mmad_dual_nfirst
// DUAL: hivm.hir.set_ctrl true at ctrl[51]
// DUAL: call @mmad_float_float_float
// DUAL-NOT: hivm.hir.set_ctrl true at ctrl[51]
// DUAL: call @mmad_float_float_float
// DUAL: hivm.hir.set_ctrl false at ctrl[51]
// DUAL-NOT: tla.mmad

// MIXED-LABEL: func.func @mmad_mixed_compute_order
// MIXED: hivm.hir.set_ctrl true at ctrl[51]
// MIXED-NOT: hivm.hir.set_ctrl true at ctrl[51]
// MIXED: call @mmad_float_float_float
// MIXED: hivm.hir.set_ctrl false at ctrl[51]
// MIXED: call @mmad_float_float_float
// MIXED: hivm.hir.set_ctrl false at ctrl[51]
// MIXED-NOT: tla.mmad
