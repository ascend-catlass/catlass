// Verify the hf32_mode attribute on tla.mmad lowers to SPR.CTRL[46] (enable)
// and SPR.CTRL[47] (rounding select), the DSL equivalents of AscendC::SetHF32Mode
// and AscendC::SetHF32TransMode.
//
// Owned by tla-insert-implicit-ctrl (InitSoc leaves CTRL[46]/[47]=0):
// HF32_NEAREST_ZERO -> set_ctrl true  at ctrl[46], set_ctrl true  at ctrl[47].
// HF32_NEAREST_EVEN -> set_ctrl true  at ctrl[46], set_ctrl false at ctrl[47].
// HF32_DISABLE      -> no insert (already matches InitSoc); epilogue still clears.
// mixed (NEAREST_ZERO + NEAREST_EVEN) -> per-mmad set_ctrl at ctrl[46]/ctrl[47].
//
// RUN: %tla_compile %s -o - | %filecheck %s --check-prefix=NZ
// RUN: %tla_compile %s -o - | %filecheck %s --check-prefix=NE
// RUN: %tla_compile %s -o - | %filecheck %s --check-prefix=DIS
// RUN: %tla_compile %s -o - | %filecheck %s --check-prefix=MIXED

module {
  tla.func @mmad_hf32_nearest_zero() {
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
      tla.mmad %11, %7, %9, %true, %c3_i64 {compute_order = #tla.compute_order<M_FIRST>, hf32_mode = #tla.hf32_mode<HF32_NEAREST_ZERO>} : !tla.tensor<!tla.layout<!tla.shape<(16,2),(16,2)>, !tla.stride<(16,256),(1,512)>, !tla.shape<32,32>, L0Clayout>, !tla.coord<0,0>, !tla.ptr<f32, l0c, 1024>>, !tla.tensor<!tla.layout<!tla.shape<(16,2),(8,4)>, !tla.stride<(8,128),(1,256)>, !tla.shape<32,32>, zN>, !tla.coord<0,0>, !tla.ptr<f32, l0a, 512>>, !tla.tensor<!tla.layout<!tla.shape<(8,4),(16,2)>, !tla.stride<(1,256),(8,128)>, !tla.shape<32,32>, nZ>, !tla.coord<0,0>, !tla.ptr<f32, l0b, 512>>, i1, i64
    }
    tla.return
  }

  tla.func @mmad_hf32_nearest_even() {
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
      tla.mmad %11, %7, %9, %true, %c3_i64 {compute_order = #tla.compute_order<M_FIRST>, hf32_mode = #tla.hf32_mode<HF32_NEAREST_EVEN>} : !tla.tensor<!tla.layout<!tla.shape<(16,2),(16,2)>, !tla.stride<(16,256),(1,512)>, !tla.shape<32,32>, L0Clayout>, !tla.coord<0,0>, !tla.ptr<f32, l0c, 1024>>, !tla.tensor<!tla.layout<!tla.shape<(16,2),(8,4)>, !tla.stride<(8,128),(1,256)>, !tla.shape<32,32>, zN>, !tla.coord<0,0>, !tla.ptr<f32, l0a, 512>>, !tla.tensor<!tla.layout<!tla.shape<(8,4),(16,2)>, !tla.stride<(1,256),(8,128)>, !tla.shape<32,32>, nZ>, !tla.coord<0,0>, !tla.ptr<f32, l0b, 512>>, i1, i64
    }
    tla.return
  }

  tla.func @mmad_hf32_disable() {
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

  tla.func @mmad_hf32_mixed() {
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
      tla.mmad %11, %7, %9, %true, %c3_i64 {compute_order = #tla.compute_order<M_FIRST>, hf32_mode = #tla.hf32_mode<HF32_NEAREST_ZERO>} : !tla.tensor<!tla.layout<!tla.shape<(16,2),(16,2)>, !tla.stride<(16,256),(1,512)>, !tla.shape<32,32>, L0Clayout>, !tla.coord<0,0>, !tla.ptr<f32, l0c, 1024>>, !tla.tensor<!tla.layout<!tla.shape<(16,2),(8,4)>, !tla.stride<(8,128),(1,256)>, !tla.shape<32,32>, zN>, !tla.coord<0,0>, !tla.ptr<f32, l0a, 512>>, !tla.tensor<!tla.layout<!tla.shape<(8,4),(16,2)>, !tla.stride<(1,256),(8,128)>, !tla.shape<32,32>, nZ>, !tla.coord<0,0>, !tla.ptr<f32, l0b, 512>>, i1, i64
      tla.mmad %11, %7, %9, %false, %c3_i64 {compute_order = #tla.compute_order<M_FIRST>, hf32_mode = #tla.hf32_mode<HF32_NEAREST_EVEN>} : !tla.tensor<!tla.layout<!tla.shape<(16,2),(16,2)>, !tla.stride<(16,256),(1,512)>, !tla.shape<32,32>, L0Clayout>, !tla.coord<0,0>, !tla.ptr<f32, l0c, 1024>>, !tla.tensor<!tla.layout<!tla.shape<(16,2),(8,4)>, !tla.stride<(8,128),(1,256)>, !tla.shape<32,32>, zN>, !tla.coord<0,0>, !tla.ptr<f32, l0a, 512>>, !tla.tensor<!tla.layout<!tla.shape<(8,4),(16,2)>, !tla.stride<(1,256),(8,128)>, !tla.shape<32,32>, nZ>, !tla.coord<0,0>, !tla.ptr<f32, l0b, 512>>, i1, i64
    }
    tla.return
  }

}

// NZ-LABEL: func.func @mmad_hf32_nearest_zero
// NZ: hivm.hir.set_ctrl true at ctrl[46]
// NZ: hivm.hir.set_ctrl true at ctrl[47]
// NZ: call @mmad_float_float_float
// NZ: hivm.hir.set_ctrl false at ctrl[46]
// NZ: hivm.hir.set_ctrl false at ctrl[47]
// NZ-NOT: tla.mmad

// NE-LABEL: func.func @mmad_hf32_nearest_even
// NE: hivm.hir.set_ctrl true at ctrl[46]
// InitSoc already leaves CTRL[47]=0; only [46] is written before the call.
// NE: call @mmad_float_float_float
// NE: hivm.hir.set_ctrl false at ctrl[46]
// NE: hivm.hir.set_ctrl false at ctrl[47]
// NE-NOT: tla.mmad

// DIS-LABEL: func.func @mmad_hf32_disable
// DIS-NOT: hivm.hir.set_ctrl true at ctrl[46]
// DIS: call @mmad_float_float_float
// DIS: hivm.hir.set_ctrl false at ctrl[46]
// DIS: hivm.hir.set_ctrl false at ctrl[47]
// DIS-NOT: tla.mmad

// MIXED-LABEL: func.func @mmad_hf32_mixed
// MIXED: hivm.hir.set_ctrl true at ctrl[46]
// MIXED: hivm.hir.set_ctrl true at ctrl[47]
// MIXED: call @mmad_float_float_float
// CTRL[46] already 1 from the first mmad; only CTRL[47] flips for NEAREST_EVEN.
// MIXED: hivm.hir.set_ctrl false at ctrl[47]
// MIXED: call @mmad_float_float_float
// MIXED: hivm.hir.set_ctrl false at ctrl[46]
// MIXED: hivm.hir.set_ctrl false at ctrl[47]
// MIXED-NOT: tla.mmad
