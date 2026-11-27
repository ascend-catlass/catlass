// RUN: %tla_compile %s -o - | %filecheck %s

!gm = !tla.tensor<!tla.layout<!tla.shape<32,32>, !tla.stride<32,1>, !tla.shape<32,32>, RowMajor>, !tla.coord<0,0>, !tla.ptr<f32, gm, 4>>
!l0a = !tla.tensor<!tla.layout<!tla.shape<(16,2),(8,4)>, !tla.stride<(8,128),(1,256)>, !tla.shape<32,32>, zN>, !tla.coord<0,0>, !tla.ptr<f8E4M3FN, l0a, 1024>>
!l0b = !tla.tensor<!tla.layout<!tla.shape<(8,4),(16,2)>, !tla.stride<(1,256),(8,128)>, !tla.shape<32,32>, nZ>, !tla.coord<0,0>, !tla.ptr<f8E4M3FN, l0b, 1024>>
!l0c = !tla.tensor<!tla.layout<!tla.shape<(16,2),(16,2)>, !tla.stride<(16,256),(1,512)>, !tla.shape<32,32>, L0Clayout>, !tla.coord<0,0>, !tla.ptr<f32, l0c, 1024>>

module {
  tla.func @fp8_mmad_ctrl(%gm: !gm) {
    %l0a_ptr = tla.alloc_ptr{size_bytes = 4096} -> !tla.ptr<f8E4M3FN, l0a, 1024>
    %l0b_ptr = tla.alloc_ptr{size_bytes = 4096} -> !tla.ptr<f8E4M3FN, l0b, 1024>
    %l0c_ptr = tla.alloc_ptr{size_bytes = 4096} -> !tla.ptr<f32, l0c, 1024>
    %l0a = tla.make_tensor_like %l0a_ptr like %gm layoutTag(#tla.layout_tag<zN>) : !tla.ptr<f8E4M3FN, l0a, 1024>, !gm -> !l0a
    %l0b = tla.make_tensor_like %l0b_ptr like %gm layoutTag(#tla.layout_tag<nZ>) : !tla.ptr<f8E4M3FN, l0b, 1024>, !gm -> !l0b
    %l0c = tla.make_tensor_like %l0c_ptr like %gm layoutTag(#tla.layout_tag<L0Clayout>) : !tla.ptr<f32, l0c, 1024>, !gm -> !l0c
    tla.cube {
      %init = arith.constant true
      %unit = arith.constant 3 : i64
      %hf8 = arith.constant 35184372088832 : i64
      tla.arch.set_ctrl %hf8 {mask = 35184372088832 : i64}
      tla.mmad_mx %l0c, %l0a, %l0b, %init, %unit {compute_order = #tla.compute_order<N_FIRST>} : !l0c, !l0a, !l0b, i1, i64
      tla.mmad %l0c, %l0a, %l0b, %init, %unit {compute_order = #tla.compute_order<M_FIRST>, hf32_mode = #tla.hf32_mode<HF32_DISABLE>} : !l0c, !l0a, !l0b, i1, i64
    }
    tla.return
  }
}

// CHECK-LABEL: func.func @fp8_mmad_ctrl
// CHECK: SET.CTRL
// CHECK-NOT: hivm.hir.set_ctrl {{.*}} at ctrl[4{{[567]}}]
// CHECK: hivm.hir.set_ctrl true at ctrl[51]
// CHECK-NEXT: call @mmadmx_fp8_e4m3fn_t_fp8_e4m3fn_t_float
// CHECK: hivm.hir.set_ctrl false at ctrl[51]
// CHECK: hivm.hir.set_ctrl false at ctrl[45]
// CHECK-NEXT: call @mmad_fp8_e4m3fn_t_fp8_e4m3fn_t_float
