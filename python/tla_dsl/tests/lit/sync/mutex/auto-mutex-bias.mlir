// RUN: split-file %s %t
// RUN: %tla_compile %t/mmad-bias.mlir -o %t/mmad-bias.o --mlir-print-ir-after=tla-insert-auto-mutex 2>&1 | %filecheck %s
// RUN: %tla_compile %t/mmad-mx-bias.mlir -o %t/mmad-mx-bias.o --mlir-print-ir-after=tla-insert-auto-mutex 2>&1 | %filecheck %s --check-prefix=MX

// The bias table is an L0-class operand of the matmul, not a copy-only tile:
// the cube reads it for the same instruction that reads L0A/L0B. auto-mutex
// therefore has to lock it on the cube side too -- the MTE1 copy into biasbuf
// being locked only on MTE1 orders the copy against itself, and orders nothing
// against the read that consumes it or the next reload that overwrites it.

//--- mmad-bias.mlir

!gm = !tla.tensor<!tla.layout<!tla.shape<32,32>, !tla.stride<32,1>, !tla.shape<32,32>, RowMajor>, !tla.coord<0,0>, !tla.ptr<f32, gm, 4>>
!gm_bias = !tla.tensor<!tla.layout<!tla.shape<32>, !tla.stride<1>, !tla.shape<32>, VectorLayout>, !tla.coord<0>, !tla.ptr<f32, gm, 4>>
!l1 = !tla.tensor<!tla.layout<!tla.shape<(16,2),(8,4)>, !tla.stride<(8,128),(1,256)>, !tla.shape<32,32>, zN>, !tla.coord<0,0>, !tla.ptr<f32, l1, 512>>
!l1_bias = !tla.tensor<!tla.layout<!tla.shape<32>, !tla.stride<1>, !tla.shape<32>, VectorLayout>, !tla.coord<0>, !tla.ptr<f32, l1, 512>>
!bias = !tla.tensor<!tla.layout<!tla.shape<32>, !tla.stride<1>, !tla.shape<32>, VectorLayout>, !tla.coord<0>, !tla.ptr<f32, biasbuf, 512>>
!l0a = !tla.tensor<!tla.layout<!tla.shape<(16,2),(8,4)>, !tla.stride<(8,128),(1,256)>, !tla.shape<32,32>, zN>, !tla.coord<0,0>, !tla.ptr<f32, l0a, 512>>
!l0b = !tla.tensor<!tla.layout<!tla.shape<(8,4),(16,2)>, !tla.stride<(1,256),(8,128)>, !tla.shape<32,32>, nZ>, !tla.coord<0,0>, !tla.ptr<f32, l0b, 512>>
!l0c = !tla.tensor<!tla.layout<!tla.shape<(16,2),(16,2)>, !tla.stride<(16,256),(1,512)>, !tla.shape<32,32>, L0Clayout>, !tla.coord<0,0>, !tla.ptr<f32, l0c, 1024>>

module {
  tla.func @auto_mutex_mmad_bias(%gm: !gm, %gm_bias: !gm_bias) attributes {tla.auto_sync = "v0"} {
    %l0c_ptr = tla.alloc_ptr{size_bytes = 4096} -> !tla.ptr<f32, l0c, 1024>
    %l0b_ptr = tla.alloc_ptr{size_bytes = 4096} -> !tla.ptr<f32, l0b, 512>
    %l0a_ptr = tla.alloc_ptr{size_bytes = 4096} -> !tla.ptr<f32, l0a, 512>
    %l1_ptr = tla.alloc_ptr{size_bytes = 4096} -> !tla.ptr<f32, l1, 512>
    %l1_bias_ptr = tla.alloc_ptr{size_bytes = 128} -> !tla.ptr<f32, l1, 512>
    %bias_ptr = tla.alloc_ptr{size_bytes = 128} -> !tla.ptr<f32, biasbuf, 512>
    %l1 = "tla.make_tensor_like"(%l1_ptr, %gm) {layoutTag = #tla.layout_tag<zN>} : (!tla.ptr<f32, l1, 512>, !gm) -> !l1
    %l0a = "tla.make_tensor_like"(%l0a_ptr, %l1) {layoutTag = #tla.layout_tag<zN>} : (!tla.ptr<f32, l0a, 512>, !l1) -> !l0a
    %l0b = "tla.make_tensor_like"(%l0b_ptr, %gm) {layoutTag = #tla.layout_tag<nZ>} : (!tla.ptr<f32, l0b, 512>, !gm) -> !l0b
    %l0c = "tla.make_tensor_like"(%l0c_ptr, %gm) {layoutTag = #tla.layout_tag<L0Clayout>} : (!tla.ptr<f32, l0c, 1024>, !gm) -> !l0c
    %l1_bias = "tla.make_tensor_like"(%l1_bias_ptr, %gm_bias) {layoutTag = #tla.layout_tag<VectorLayout>} : (!tla.ptr<f32, l1, 512>, !gm_bias) -> !l1_bias
    %bias = "tla.make_tensor_like"(%bias_ptr, %gm_bias) {layoutTag = #tla.layout_tag<VectorLayout>} : (!tla.ptr<f32, biasbuf, 512>, !gm_bias) -> !bias
    %init = arith.constant true
    %unit = arith.constant 0 : i64
    "tla.cube"() ({
      tla.copy %l1_bias, %gm_bias : !l1_bias, !gm_bias
      tla.copy %bias, %l1_bias : !bias, !l1_bias
      tla.copy %l1, %gm : !l1, !gm
      tla.copy %l0a, %l1 : !l0a, !l1
      tla.mmad %l0c, %l0a, %l0b, %init, %unit, %bias {compute_order = #tla.compute_order<M_FIRST>, hf32_mode = #tla.hf32_mode<HF32_DISABLE>} : !l0c, !l0a, !l0b, i1, i64, !bias
    }) : () -> ()
    tla.return
  }
}

// Biasbuf gets its own ID, ranked after L0C (biasbuf is an L0-class table).
// CHECK-LABEL: func.func @auto_mutex_mmad_bias
// CHECK-NOT: tla.auto_sync
// CHECK: %[[L1:.*]] = tla.mutex "auto_l1_0_4096" {id = 0 : i64} -> !tla.mutex
// CHECK-NEXT: %[[L1_BIAS:.*]] = tla.mutex "auto_l1_4096_128" {id = 1 : i64} -> !tla.mutex
// CHECK-NEXT: %[[L0A:.*]] = tla.mutex "auto_l0a_0_4096" {id = 2 : i64} -> !tla.mutex
// CHECK-NEXT: %[[L0B:.*]] = tla.mutex "auto_l0b_0_4096" {id = 3 : i64} -> !tla.mutex
// CHECK-NEXT: %[[L0C:.*]] = tla.mutex "auto_l0c_0_4096" {id = 4 : i64} -> !tla.mutex
// CHECK-NEXT: %[[BIASBUF:.*]] = tla.mutex "auto_biasbuf_0_128" {id = 5 : i64} -> !tla.mutex
// The MTE1 copy fills biasbuf under its mutex ...
// CHECK: tla.mutex_lock %[[L1_BIAS]][<mte1>] : !tla.mutex
// CHECK-NEXT: tla.mutex_lock %[[BIASBUF]][<mte1>] : !tla.mutex
// CHECK-NEXT: tla.copy
// CHECK-NEXT: tla.mutex_unlock %[[BIASBUF]][<mte1>] : !tla.mutex
// CHECK-NEXT: tla.mutex_unlock %[[L1_BIAS]][<mte1>] : !tla.mutex
// ... and the MMAD takes the same mutex on the cube pipe, exactly like L0A/L0B,
// so the read is ordered after the fill and the next fill after the read.
// CHECK: tla.mutex_lock %[[L0A]][<cube>] : !tla.mutex
// CHECK-NEXT: tla.mutex_lock %[[L0B]][<cube>] : !tla.mutex
// CHECK-NEXT: tla.mutex_lock %[[L0C]][<cube>] : !tla.mutex
// CHECK-NEXT: tla.mutex_lock %[[BIASBUF]][<cube>] : !tla.mutex
// CHECK-NEXT: tla.mmad
// CHECK-NEXT: tla.mutex_unlock %[[BIASBUF]][<cube>] : !tla.mutex
// CHECK-NEXT: tla.mutex_unlock %[[L0C]][<cube>] : !tla.mutex
// CHECK-NEXT: tla.mutex_unlock %[[L0B]][<cube>] : !tla.mutex
// CHECK-NEXT: tla.mutex_unlock %[[L0A]][<cube>] : !tla.mutex

//--- mmad-mx-bias.mlir

!mx_gm = !tla.tensor<!tla.layout<!tla.shape<32,32>, !tla.stride<32,1>, !tla.shape<32,32>, RowMajor>, !tla.coord<0,0>, !tla.ptr<f32, gm, 4>>
!mx_gm_bias = !tla.tensor<!tla.layout<!tla.shape<32>, !tla.stride<1>, !tla.shape<32>, VectorLayout>, !tla.coord<0>, !tla.ptr<f32, gm, 4>>
!mx_l0a = !tla.tensor<!tla.layout<!tla.shape<(16,2),(8,4)>, !tla.stride<(8,128),(1,256)>, !tla.shape<32,32>, zN>, !tla.coord<0,0>, !tla.ptr<f8E4M3FN, l0a, 1024>>
!mx_l0b = !tla.tensor<!tla.layout<!tla.shape<(8,4),(16,2)>, !tla.stride<(1,256),(8,128)>, !tla.shape<32,32>, nZ>, !tla.coord<0,0>, !tla.ptr<f8E4M3FN, l0b, 1024>>
!mx_l0c = !tla.tensor<!tla.layout<!tla.shape<(16,2),(16,2)>, !tla.stride<(16,256),(1,512)>, !tla.shape<32,32>, L0Clayout>, !tla.coord<0,0>, !tla.ptr<f32, l0c, 1024>>
!mx_bias = !tla.tensor<!tla.layout<!tla.shape<32>, !tla.stride<1>, !tla.shape<32>, VectorLayout>, !tla.coord<0>, !tla.ptr<f32, biasbuf, 512>>

module {
  tla.func @auto_mutex_mmad_mx_bias(%gm: !mx_gm, %gm_bias: !mx_gm_bias) attributes {tla.auto_sync = "v0"} {
    %l0c_ptr = tla.alloc_ptr{size_bytes = 4096} -> !tla.ptr<f32, l0c, 1024>
    %l0b_ptr = tla.alloc_ptr{size_bytes = 4096} -> !tla.ptr<f8E4M3FN, l0b, 1024>
    %l0a_ptr = tla.alloc_ptr{size_bytes = 4096} -> !tla.ptr<f8E4M3FN, l0a, 1024>
    %bias_ptr = tla.alloc_ptr{size_bytes = 128} -> !tla.ptr<f32, biasbuf, 512>
    %l0a = "tla.make_tensor_like"(%l0a_ptr, %gm) {layoutTag = #tla.layout_tag<zN>} : (!tla.ptr<f8E4M3FN, l0a, 1024>, !mx_gm) -> !mx_l0a
    %l0b = "tla.make_tensor_like"(%l0b_ptr, %gm) {layoutTag = #tla.layout_tag<nZ>} : (!tla.ptr<f8E4M3FN, l0b, 1024>, !mx_gm) -> !mx_l0b
    %l0c = "tla.make_tensor_like"(%l0c_ptr, %gm) {layoutTag = #tla.layout_tag<L0Clayout>} : (!tla.ptr<f32, l0c, 1024>, !mx_gm) -> !mx_l0c
    %bias = "tla.make_tensor_like"(%bias_ptr, %gm_bias) {layoutTag = #tla.layout_tag<VectorLayout>} : (!tla.ptr<f32, biasbuf, 512>, !mx_gm_bias) -> !mx_bias
    %init = arith.constant true
    %unit = arith.constant 3 : i64
    "tla.cube"() ({
      tla.mmad_mx %l0c, %l0a, %l0b, %init, %unit, %bias {compute_order = #tla.compute_order<M_FIRST>} : !mx_l0c, !mx_l0a, !mx_l0b, i1, i64, !mx_bias
    }) : () -> ()
    tla.return
  }
}

// The MX flavour takes the same resources. unit_flag 3 removes L0C from the
// mutex set -- the unit-flag protocol owns it -- but must not remove the bias,
// which the flag never covers.
// MX-LABEL: func.func @auto_mutex_mmad_mx_bias
// MX-NOT: tla.auto_sync
// MX: %[[MX_L0A:.*]] = tla.mutex "auto_l0a_0_4096" {id = 0 : i64} -> !tla.mutex
// MX-NEXT: %[[MX_L0B:.*]] = tla.mutex "auto_l0b_0_4096" {id = 1 : i64} -> !tla.mutex
// MX-NEXT: %[[MX_BIAS:.*]] = tla.mutex "auto_biasbuf_0_128" {id = 2 : i64} -> !tla.mutex
// MX: tla.mutex_lock %[[MX_L0A]][<cube>] : !tla.mutex
// MX-NEXT: tla.mutex_lock %[[MX_L0B]][<cube>] : !tla.mutex
// MX-NEXT: tla.mutex_lock %[[MX_BIAS]][<cube>] : !tla.mutex
// MX-NEXT: tla.mmad_mx
// MX-NEXT: tla.mutex_unlock %[[MX_BIAS]][<cube>] : !tla.mutex
// MX-NEXT: tla.mutex_unlock %[[MX_L0B]][<cube>] : !tla.mutex
// MX-NEXT: tla.mutex_unlock %[[MX_L0A]][<cube>] : !tla.mutex
