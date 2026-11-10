// Control-flow regressions for tla-insert-implicit-ctrl (keep minimal):
// - scf.if Mixed arms fork from the same entry state.
// - Mixed for hoists Uniform bits (zero-trip safe); Mixed bits stay inside.
//
// RUN: %tla_compile %s -o /dev/null --mlir-print-ir-after=tla-insert-implicit-ctrl 2>&1 | %filecheck %s --check-prefix=IF
// RUN: %tla_compile %s -o /dev/null --mlir-print-ir-after=tla-insert-implicit-ctrl 2>&1 | %filecheck %s --check-prefix=POST

!l1 = !tla.tensor<!tla.layout<!tla.shape<32,32>, !tla.stride<32,1>, !tla.shape<32,32>, RowMajor>, !tla.coord<0,0>, !tla.ptr<f32, l1, 512>>
!l0a = !tla.tensor<!tla.layout<!tla.shape<(16,2),(8,4)>, !tla.stride<(8,128),(1,256)>, !tla.shape<32,32>, zN>, !tla.coord<0,0>, !tla.ptr<f32, l0a, 512>>
!l0b = !tla.tensor<!tla.layout<!tla.shape<(8,4),(16,2)>, !tla.stride<(1,256),(8,128)>, !tla.shape<32,32>, nZ>, !tla.coord<0,0>, !tla.ptr<f32, l0b, 512>>
!l0c = !tla.tensor<!tla.layout<!tla.shape<(16,2),(16,2)>, !tla.stride<(16,256),(1,512)>, !tla.shape<32,32>, L0Clayout>, !tla.coord<0,0>, !tla.ptr<f32, l0c, 512>>

module {

// then: NEAREST_ZERO; else: NEAREST_EVEN — else must still set [46] from entry.
// IF-LABEL: func.func @mmad_hf32_if_mixed
// IF: scf.if
// IF: hivm.hir.set_ctrl true at ctrl[46]
// IF: hivm.hir.set_ctrl true at ctrl[47]
// IF: tla.mmad
// IF: } else {
// IF: hivm.hir.set_ctrl true at ctrl[46]
// IF: tla.mmad
  tla.func @mmad_hf32_if_mixed(%cond: i1) {
    %0 = tla.alloc_ptr{size_bytes = 4096} -> !tla.ptr<f32, l1, 512>
    %1 = tla.make_shape -> !tla.shape<32,32>
    %2 = tla.make_stride -> !tla.stride<32,1>
    %3 = tla.make_layout %1, %2 : !tla.shape<32,32>, !tla.stride<32,1> -> !tla.layout<!tla.shape<32,32>, !tla.stride<32,1>, !tla.shape<32,32>, RowMajor>
    %4 = tla.make_coord -> !tla.coord<0,0>
    %5 = tla.make_tensor %0, %3, %4 : !tla.ptr<f32, l1, 512>, !tla.layout<!tla.shape<32,32>, !tla.stride<32,1>, !tla.shape<32,32>, RowMajor>, !tla.coord<0,0> -> !l1
    %6 = tla.alloc_ptr{size_bytes = 4096} -> !tla.ptr<f32, l0a, 512>
    %7 = tla.make_tensor_like %6 like %5 layoutTag(#tla.layout_tag<zN>) : !tla.ptr<f32, l0a, 512>, !l1 -> !l0a
    %8 = tla.alloc_ptr{size_bytes = 4096} -> !tla.ptr<f32, l0b, 512>
    %9 = tla.make_tensor_like %8 like %5 layoutTag(#tla.layout_tag<nZ>) : !tla.ptr<f32, l0b, 512>, !l1 -> !l0b
    %10 = tla.alloc_ptr{size_bytes = 4096} -> !tla.ptr<f32, l0c, 512>
    %11 = tla.make_tensor_like %10 like %5 layoutTag(#tla.layout_tag<L0Clayout>) : !tla.ptr<f32, l0c, 512>, !l1 -> !l0c
    tla.cube {
      %true = arith.constant true
      %c3_i64 = arith.constant 3 : i64
      scf.if %cond {
        tla.mmad %11, %7, %9, %true, %c3_i64 {compute_order = #tla.compute_order<M_FIRST>, hf32_mode = #tla.hf32_mode<HF32_NEAREST_ZERO>} : !l0c, !l0a, !l0b, i1, i64
      } else {
        tla.mmad %11, %7, %9, %true, %c3_i64 {compute_order = #tla.compute_order<M_FIRST>, hf32_mode = #tla.hf32_mode<HF32_NEAREST_EVEN>} : !l0c, !l0a, !l0b, i1, i64
      }
    }
    tla.return
  }

// Mixed NZ+NE: hoist uniform [46]; [47] Mixed inside; post-loop NE still correct after zero-trip.
// POST-LABEL: func.func @mmad_hf32_for_mixed_then_post
// POST: hivm.hir.set_ctrl true at ctrl[46]
// POST: scf.for
// POST: hivm.hir.set_ctrl true at ctrl[47]
// POST: tla.mmad
// POST: hivm.hir.set_ctrl false at ctrl[47]
// POST: tla.mmad
// POST: hivm.hir.set_ctrl false at ctrl[47]
// POST: tla.mmad
  tla.func @mmad_hf32_for_mixed_then_post(%lb: index, %ub: index, %step: index) {
    %0 = tla.alloc_ptr{size_bytes = 4096} -> !tla.ptr<f32, l1, 512>
    %1 = tla.make_shape -> !tla.shape<32,32>
    %2 = tla.make_stride -> !tla.stride<32,1>
    %3 = tla.make_layout %1, %2 : !tla.shape<32,32>, !tla.stride<32,1> -> !tla.layout<!tla.shape<32,32>, !tla.stride<32,1>, !tla.shape<32,32>, RowMajor>
    %4 = tla.make_coord -> !tla.coord<0,0>
    %5 = tla.make_tensor %0, %3, %4 : !tla.ptr<f32, l1, 512>, !tla.layout<!tla.shape<32,32>, !tla.stride<32,1>, !tla.shape<32,32>, RowMajor>, !tla.coord<0,0> -> !l1
    %6 = tla.alloc_ptr{size_bytes = 4096} -> !tla.ptr<f32, l0a, 512>
    %7 = tla.make_tensor_like %6 like %5 layoutTag(#tla.layout_tag<zN>) : !tla.ptr<f32, l0a, 512>, !l1 -> !l0a
    %8 = tla.alloc_ptr{size_bytes = 4096} -> !tla.ptr<f32, l0b, 512>
    %9 = tla.make_tensor_like %8 like %5 layoutTag(#tla.layout_tag<nZ>) : !tla.ptr<f32, l0b, 512>, !l1 -> !l0b
    %10 = tla.alloc_ptr{size_bytes = 4096} -> !tla.ptr<f32, l0c, 512>
    %11 = tla.make_tensor_like %10 like %5 layoutTag(#tla.layout_tag<L0Clayout>) : !tla.ptr<f32, l0c, 512>, !l1 -> !l0c
    tla.cube {
      %true = arith.constant true
      %false = arith.constant false
      %c3_i64 = arith.constant 3 : i64
      scf.for %i = %lb to %ub step %step {
        tla.mmad %11, %7, %9, %true, %c3_i64 {compute_order = #tla.compute_order<M_FIRST>, hf32_mode = #tla.hf32_mode<HF32_NEAREST_ZERO>} : !l0c, !l0a, !l0b, i1, i64
        tla.mmad %11, %7, %9, %false, %c3_i64 {compute_order = #tla.compute_order<M_FIRST>, hf32_mode = #tla.hf32_mode<HF32_NEAREST_EVEN>} : !l0c, !l0a, !l0b, i1, i64
      }
      tla.mmad %11, %7, %9, %true, %c3_i64 {compute_order = #tla.compute_order<M_FIRST>, hf32_mode = #tla.hf32_mode<HF32_NEAREST_EVEN>} : !l0c, !l0a, !l0b, i1, i64
    }
    tla.return
  }
}
