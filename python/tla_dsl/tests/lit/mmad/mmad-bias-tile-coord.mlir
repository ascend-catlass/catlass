// A bias tile carved out of a larger buffer must reach tla.mmad at its own
// offset, not at element 0 of the buffer.
//
// The copy routes carry the tile coordinate in the descriptor payload and let
// the bc side compute the address, but tla.mmad hands the bias memref straight
// to a wrapper that reads `bias->aligned + bias->offset` and never looks at a
// coordinate. So a rank-1 tile whose coord is non-zero has to be materialised
// as a subview of its backing buffer -- otherwise the matmul would consume the
// head of the buffer instead of the slice the copy filled.
//
// The second function is the control: the same tile at coord 0 keeps the whole
// buffer, so the subview is not emitted unconditionally.

// RUN: split-file %s %t
// RUN: %tla_compile %t/offset.mlir -o %t/offset.o --mlir-print-ir-after=tla-cube-region 2>&1 | %filecheck %s
// RUN: %tla_compile %t/zero.mlir -o %t/zero.o --mlir-print-ir-after=tla-cube-region 2>&1 | %filecheck %s --check-prefix=ZERO

//--- offset.mlir

!gm = !tla.tensor<!tla.layout<!tla.shape<32,32>, !tla.stride<32,1>, !tla.shape<32,32>, RowMajor>, !tla.coord<0,0>, !tla.ptr<f32, gm, 4>>
!gm_bias = !tla.tensor<!tla.layout<!tla.shape<32>, !tla.stride<1>, !tla.shape<32>, VectorLayout>, !tla.coord<0>, !tla.ptr<f32, gm, 4>>
!l1_bias = !tla.tensor<!tla.layout<!tla.shape<32>, !tla.stride<1>, !tla.shape<32>, VectorLayout>, !tla.coord<0>, !tla.ptr<f32, l1, 512>>
!bias_tile = !tla.tensor<!tla.layout<!tla.shape<32>, !tla.stride<1>, !tla.shape<32>, VectorLayout>, !tla.coord<512>, !tla.ptr<f32, biasbuf, 512>>
!l1 = !tla.tensor<!tla.layout<!tla.shape<(16,2),(8,4)>, !tla.stride<(8,128),(1,256)>, !tla.shape<32,32>, zN>, !tla.coord<0,0>, !tla.ptr<f32, l1, 512>>
!l0a = !tla.tensor<!tla.layout<!tla.shape<(16,2),(8,4)>, !tla.stride<(8,128),(1,256)>, !tla.shape<32,32>, zN>, !tla.coord<0,0>, !tla.ptr<f32, l0a, 512>>
!l0b = !tla.tensor<!tla.layout<!tla.shape<(8,4),(16,2)>, !tla.stride<(1,256),(8,128)>, !tla.shape<32,32>, nZ>, !tla.coord<0,0>, !tla.ptr<f32, l0b, 512>>
!l0c = !tla.tensor<!tla.layout<!tla.shape<(16,2),(16,2)>, !tla.stride<(16,256),(1,512)>, !tla.shape<32,32>, L0Clayout>, !tla.coord<0,0>, !tla.ptr<f32, l0c, 1024>>

module {
  tla.func @bias_tile_offset(%gm: !gm, %gm_bias: !gm_bias) {
    %l0c_ptr = tla.alloc_ptr{size_bytes = 4096} -> !tla.ptr<f32, l0c, 1024>
    %l0b_ptr = tla.alloc_ptr{size_bytes = 4096} -> !tla.ptr<f32, l0b, 512>
    %l0a_ptr = tla.alloc_ptr{size_bytes = 4096} -> !tla.ptr<f32, l0a, 512>
    %l1_ptr = tla.alloc_ptr{size_bytes = 4096} -> !tla.ptr<f32, l1, 512>
    %l1_bias_ptr = tla.alloc_ptr{size_bytes = 128} -> !tla.ptr<f32, l1, 512>
    // 1024 f32 = 4 KiB: the bias table buffer is deliberately larger than the
    // 32-element tile the matmul consumes.
    %bias_ptr = tla.alloc_ptr{size_bytes = 4096} -> !tla.ptr<f32, biasbuf, 512>
    %l1 = "tla.make_tensor_like"(%l1_ptr, %gm) {layoutTag = #tla.layout_tag<zN>} : (!tla.ptr<f32, l1, 512>, !gm) -> !l1
    %l0a = "tla.make_tensor_like"(%l0a_ptr, %l1) {layoutTag = #tla.layout_tag<zN>} : (!tla.ptr<f32, l0a, 512>, !l1) -> !l0a
    %l0b = "tla.make_tensor_like"(%l0b_ptr, %gm) {layoutTag = #tla.layout_tag<nZ>} : (!tla.ptr<f32, l0b, 512>, !gm) -> !l0b
    %l0c = "tla.make_tensor_like"(%l0c_ptr, %gm) {layoutTag = #tla.layout_tag<L0Clayout>} : (!tla.ptr<f32, l0c, 1024>, !gm) -> !l0c
    %l1_bias = "tla.make_tensor_like"(%l1_bias_ptr, %gm_bias) {layoutTag = #tla.layout_tag<VectorLayout>} : (!tla.ptr<f32, l1, 512>, !gm_bias) -> !l1_bias
    // The tile the matmul consumes starts at element 512 of that buffer. It is
    // spelled as its own tensor over the same pointer rather than as a
    // tile_view, so the coordinate reaches the descriptor unchanged.
    %bshape = "tla.make_shape"() : () -> !tla.shape<32>
    %bstride = "tla.make_stride"() : () -> !tla.stride<1>
    %blayout = "tla.make_layout"(%bshape, %bstride) : (!tla.shape<32>, !tla.stride<1>) -> !tla.layout<!tla.shape<32>, !tla.stride<1>, !tla.shape<32>, VectorLayout>
    %bcoord = "tla.make_coord"() : () -> !tla.coord<512>
    %bias_tile = "tla.make_tensor"(%bias_ptr, %blayout, %bcoord) : (!tla.ptr<f32, biasbuf, 512>, !tla.layout<!tla.shape<32>, !tla.stride<1>, !tla.shape<32>, VectorLayout>, !tla.coord<512>) -> !bias_tile
    %init = arith.constant true
    %unit = arith.constant 0 : i64
    "tla.cube"() ({
      tla.copy %l1_bias, %gm_bias : !l1_bias, !gm_bias
      tla.copy %bias_tile, %l1_bias : !bias_tile, !l1_bias
      tla.copy %l1, %gm : !l1, !gm
      tla.copy %l0a, %l1 : !l0a, !l1
      tla.mmad %l0c, %l0a, %l0b, %init, %unit, %bias_tile {compute_order = #tla.compute_order<M_FIRST>, hf32_mode = #tla.hf32_mode<HF32_DISABLE>} : !l0c, !l0a, !l0b, i1, i64, !bias_tile
    }) : () -> ()
    tla.return
  }
}

// CHECK-LABEL: func.func @bias_tile_offset
// The bias the matmul receives starts at element 512 of the 1024-element
// buffer, so it is a subview of it rather than the whole allocation.
// CHECK: %[[BIAS:.*]] = memref.subview %{{.*}}[%{{.*}}] [%{{.*}}] [%{{.*}}] : memref<1024xf32, #hivm.address_space<biasbuf>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<biasbuf>>
// CHECK: call @mmad_bias_float_float_float({{.*}}, %[[BIAS]]) : ({{.*}}, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<biasbuf>>) -> ()

//--- zero.mlir

!zgm = !tla.tensor<!tla.layout<!tla.shape<32,32>, !tla.stride<32,1>, !tla.shape<32,32>, RowMajor>, !tla.coord<0,0>, !tla.ptr<f32, gm, 4>>
!zgm_bias = !tla.tensor<!tla.layout<!tla.shape<32>, !tla.stride<1>, !tla.shape<32>, VectorLayout>, !tla.coord<0>, !tla.ptr<f32, gm, 4>>
!zl1_bias = !tla.tensor<!tla.layout<!tla.shape<32>, !tla.stride<1>, !tla.shape<32>, VectorLayout>, !tla.coord<0>, !tla.ptr<f32, l1, 512>>
!zbias = !tla.tensor<!tla.layout<!tla.shape<32>, !tla.stride<1>, !tla.shape<32>, VectorLayout>, !tla.coord<0>, !tla.ptr<f32, biasbuf, 512>>
!zl1 = !tla.tensor<!tla.layout<!tla.shape<(16,2),(8,4)>, !tla.stride<(8,128),(1,256)>, !tla.shape<32,32>, zN>, !tla.coord<0,0>, !tla.ptr<f32, l1, 512>>
!zl0a = !tla.tensor<!tla.layout<!tla.shape<(16,2),(8,4)>, !tla.stride<(8,128),(1,256)>, !tla.shape<32,32>, zN>, !tla.coord<0,0>, !tla.ptr<f32, l0a, 512>>
!zl0b = !tla.tensor<!tla.layout<!tla.shape<(8,4),(16,2)>, !tla.stride<(1,256),(8,128)>, !tla.shape<32,32>, nZ>, !tla.coord<0,0>, !tla.ptr<f32, l0b, 512>>
!zl0c = !tla.tensor<!tla.layout<!tla.shape<(16,2),(16,2)>, !tla.stride<(16,256),(1,512)>, !tla.shape<32,32>, L0Clayout>, !tla.coord<0,0>, !tla.ptr<f32, l0c, 1024>>

module {
  tla.func @bias_tile_zero(%gm: !zgm, %gm_bias: !zgm_bias) {
    %l0c_ptr = tla.alloc_ptr{size_bytes = 4096} -> !tla.ptr<f32, l0c, 1024>
    %l0b_ptr = tla.alloc_ptr{size_bytes = 4096} -> !tla.ptr<f32, l0b, 512>
    %l0a_ptr = tla.alloc_ptr{size_bytes = 4096} -> !tla.ptr<f32, l0a, 512>
    %l1_ptr = tla.alloc_ptr{size_bytes = 4096} -> !tla.ptr<f32, l1, 512>
    %l1_bias_ptr = tla.alloc_ptr{size_bytes = 128} -> !tla.ptr<f32, l1, 512>
    %bias_ptr = tla.alloc_ptr{size_bytes = 128} -> !tla.ptr<f32, biasbuf, 512>
    %l1 = "tla.make_tensor_like"(%l1_ptr, %gm) {layoutTag = #tla.layout_tag<zN>} : (!tla.ptr<f32, l1, 512>, !zgm) -> !zl1
    %l0a = "tla.make_tensor_like"(%l0a_ptr, %l1) {layoutTag = #tla.layout_tag<zN>} : (!tla.ptr<f32, l0a, 512>, !zl1) -> !zl0a
    %l0b = "tla.make_tensor_like"(%l0b_ptr, %gm) {layoutTag = #tla.layout_tag<nZ>} : (!tla.ptr<f32, l0b, 512>, !zgm) -> !zl0b
    %l0c = "tla.make_tensor_like"(%l0c_ptr, %gm) {layoutTag = #tla.layout_tag<L0Clayout>} : (!tla.ptr<f32, l0c, 1024>, !zgm) -> !zl0c
    %l1_bias = "tla.make_tensor_like"(%l1_bias_ptr, %gm_bias) {layoutTag = #tla.layout_tag<VectorLayout>} : (!tla.ptr<f32, l1, 512>, !zgm_bias) -> !zl1_bias
    %bias = "tla.make_tensor_like"(%bias_ptr, %gm_bias) {layoutTag = #tla.layout_tag<VectorLayout>} : (!tla.ptr<f32, biasbuf, 512>, !zgm_bias) -> !zbias
    %init = arith.constant true
    %unit = arith.constant 0 : i64
    "tla.cube"() ({
      tla.copy %l1_bias, %gm_bias : !zl1_bias, !zgm_bias
      tla.copy %bias, %l1_bias : !zbias, !zl1_bias
      tla.copy %l1, %gm : !zl1, !zgm
      tla.copy %l0a, %l1 : !zl0a, !zl1
      tla.mmad %l0c, %l0a, %l0b, %init, %unit, %bias {compute_order = #tla.compute_order<M_FIRST>, hf32_mode = #tla.hf32_mode<HF32_DISABLE>} : !zl0c, !zl0a, !zl0b, i1, i64, !zbias
    }) : () -> ()
    tla.return
  }
}

// A coord of zero needs no subview: the whole buffer already starts at the
// tile. This keeps the common case free of an extra op.
// ZERO-LABEL: func.func @bias_tile_zero
// ZERO-NOT: memref.subview
// ZERO: call @mmad_bias_
