// RUN: %tla_compile %s -o - | %filecheck %s
//
// GEMV form of tla.mmad: M is 1, so the lhs tile is a 1-D VectorLayout row
// instead of a zN matrix tile, while the rhs keeps the nZ fractal block and the
// accumulator keeps L0Clayout. The layout -- not a separate attribute -- is what
// selects the mode, and the pass derives disableGemv from it for the bc wrapper.
//
// Every leg of the data path is the vector-to-vector route: GM -> L1 and
// L1 -> L0A both move VectorLayout to VectorLayout for the A row, and the B
// block moves nZ to nZ. The accumulator still leaves through the ordinary
// RowMajor fixpipe, which is what writes the GEMV result as a row.

module attributes {tla.module_exec_units = "cube"} {
  tla.func @gemv_mmad(
      %arg_a: !tla.tensor<!tla.layout<!tla.shape<32>, !tla.stride<1>, !tla.shape<32>, VectorLayout>, !tla.coord<0>, !tla.ptr<f32, gm, 4>>,
      %arg_b: !tla.tensor<!tla.layout<!tla.shape<32,32>, !tla.stride<32,1>, !tla.shape<32,32>, RowMajor>, !tla.coord<0,0>, !tla.ptr<f32, gm, 4>>,
      %arg_c: !tla.tensor<!tla.layout<!tla.shape<1,32>, !tla.stride<32,1>, !tla.shape<1,32>, RowMajor>, !tla.coord<0,0>, !tla.ptr<f32, gm, 4>>) {
    // A row: VectorLayout all the way from GM to L0A.
    %l1a = tla.alloc_ptr{size_bytes = 8192} -> !tla.ptr<f32, l1, 512>
    %l1_a = tla.make_tensor_like %l1a like %arg_a layoutTag(#tla.layout_tag<VectorLayout>) :
      !tla.ptr<f32, l1, 512>,
      !tla.tensor<!tla.layout<!tla.shape<32>, !tla.stride<1>, !tla.shape<32>, VectorLayout>, !tla.coord<0>, !tla.ptr<f32, gm, 4>> ->
      !tla.tensor<!tla.layout<!tla.shape<32>, !tla.stride<1>, !tla.shape<32>, VectorLayout>, !tla.coord<0>, !tla.ptr<f32, l1, 512>>
    %l0a = tla.alloc_ptr{size_bytes = 4096} -> !tla.ptr<f32, l0a, 512>
    %l0_a = tla.make_tensor_like %l0a like %l1_a layoutTag(#tla.layout_tag<VectorLayout>) :
      !tla.ptr<f32, l0a, 512>,
      !tla.tensor<!tla.layout<!tla.shape<32>, !tla.stride<1>, !tla.shape<32>, VectorLayout>, !tla.coord<0>, !tla.ptr<f32, l1, 512>> ->
      !tla.tensor<!tla.layout<!tla.shape<32>, !tla.stride<1>, !tla.shape<32>, VectorLayout>, !tla.coord<0>, !tla.ptr<f32, l0a, 512>>

    // B block: a prepacked nZ weight buffer reached through tensor_ptr.
    %b_tile_shape = tla.make_shape -> !tla.shape<32,32>
    %b_tile_coord = tla.make_coord -> !tla.coord<0,0>
    %b_tile = tla.tile_view %arg_b, %b_tile_shape, %b_tile_coord :
      !tla.tensor<!tla.layout<!tla.shape<32,32>, !tla.stride<32,1>, !tla.shape<32,32>, RowMajor>, !tla.coord<0,0>, !tla.ptr<f32, gm, 4>>,
      !tla.shape<32,32>, !tla.coord<0,0> ->
      !tla.tensor<!tla.layout<!tla.shape<32,32>, !tla.stride<32,1>, !tla.shape<32,32>, RowMajor>, !tla.coord<0,0>, !tla.ptr<f32, gm, 4>>
    %b_ptr = tla.tensor_ptr %b_tile :
      !tla.tensor<!tla.layout<!tla.shape<32,32>, !tla.stride<32,1>, !tla.shape<32,32>, RowMajor>, !tla.coord<0,0>, !tla.ptr<f32, gm, 4>> ->
      !tla.ptr<f32, gm, 4>
    %b_shape = tla.make_shape -> !tla.shape<(8,4),(16,2)>
    %b_stride = tla.make_stride -> !tla.stride<(1,256),(8,128)>
    %b_origin = tla.make_shape -> !tla.shape<32,32>
    %b_layout = tla.make_layout %b_shape, %b_stride origin %b_origin {layoutTag = #tla.layout_tag<nZ>} :
      !tla.shape<(8,4),(16,2)>, !tla.stride<(1,256),(8,128)> origin !tla.shape<32,32> ->
      !tla.layout<!tla.shape<(8,4),(16,2)>, !tla.stride<(1,256),(8,128)>, !tla.shape<32,32>, nZ>
    %b_coord = tla.make_coord -> !tla.coord<0,0>
    %b_gm = tla.make_tensor %b_ptr, %b_layout, %b_coord :
      !tla.ptr<f32, gm, 4>,
      !tla.layout<!tla.shape<(8,4),(16,2)>, !tla.stride<(1,256),(8,128)>, !tla.shape<32,32>, nZ>,
      !tla.coord<0,0> ->
      !tla.tensor<!tla.layout<!tla.shape<(8,4),(16,2)>, !tla.stride<(1,256),(8,128)>, !tla.shape<32,32>, nZ>, !tla.coord<0,0>, !tla.ptr<f32, gm, 4>>
    %l1b = tla.alloc_ptr{size_bytes = 8192} -> !tla.ptr<f32, l1, 512>
    %l1_b = tla.make_tensor_like %l1b like %b_gm layoutTag(#tla.layout_tag<nZ>) :
      !tla.ptr<f32, l1, 512>,
      !tla.tensor<!tla.layout<!tla.shape<(8,4),(16,2)>, !tla.stride<(1,256),(8,128)>, !tla.shape<32,32>, nZ>, !tla.coord<0,0>, !tla.ptr<f32, gm, 4>> ->
      !tla.tensor<!tla.layout<!tla.shape<(8,4),(16,2)>, !tla.stride<(1,256),(8,128)>, !tla.shape<32,32>, nZ>, !tla.coord<0,0>, !tla.ptr<f32, l1, 512>>
    %l0b = tla.alloc_ptr{size_bytes = 4096} -> !tla.ptr<f32, l0b, 512>
    %l0_b = tla.make_tensor_like %l0b like %l1_b layoutTag(#tla.layout_tag<nZ>) :
      !tla.ptr<f32, l0b, 512>,
      !tla.tensor<!tla.layout<!tla.shape<(8,4),(16,2)>, !tla.stride<(1,256),(8,128)>, !tla.shape<32,32>, nZ>, !tla.coord<0,0>, !tla.ptr<f32, l1, 512>> ->
      !tla.tensor<!tla.layout<!tla.shape<(8,4),(16,2)>, !tla.stride<(1,256),(8,128)>, !tla.shape<32,32>, nZ>, !tla.coord<0,0>, !tla.ptr<f32, l0b, 512>>

    // Accumulator: still L0Clayout, an M=1 row of N.
    %l0c = tla.alloc_ptr{size_bytes = 4096} -> !tla.ptr<f32, l0c, 1024>
    %l0_c = tla.make_tensor_like %l0c like %arg_c layoutTag(#tla.layout_tag<L0Clayout>) :
      !tla.ptr<f32, l0c, 1024>,
      !tla.tensor<!tla.layout<!tla.shape<1,32>, !tla.stride<32,1>, !tla.shape<1,32>, RowMajor>, !tla.coord<0,0>, !tla.ptr<f32, gm, 4>> ->
      !tla.tensor<!tla.layout<!tla.shape<(16,1),(16,2)>, !tla.stride<(16,256),(1,256)>, !tla.shape<1,32>, L0Clayout>, !tla.coord<0,0>, !tla.ptr<f32, l0c, 1024>>

    tla.cube {
      tla.copy %l1_a, %arg_a :
        !tla.tensor<!tla.layout<!tla.shape<32>, !tla.stride<1>, !tla.shape<32>, VectorLayout>, !tla.coord<0>, !tla.ptr<f32, l1, 512>>,
        !tla.tensor<!tla.layout<!tla.shape<32>, !tla.stride<1>, !tla.shape<32>, VectorLayout>, !tla.coord<0>, !tla.ptr<f32, gm, 4>>
      tla.copy %l1_b, %b_gm :
        !tla.tensor<!tla.layout<!tla.shape<(8,4),(16,2)>, !tla.stride<(1,256),(8,128)>, !tla.shape<32,32>, nZ>, !tla.coord<0,0>, !tla.ptr<f32, l1, 512>>,
        !tla.tensor<!tla.layout<!tla.shape<(8,4),(16,2)>, !tla.stride<(1,256),(8,128)>, !tla.shape<32,32>, nZ>, !tla.coord<0,0>, !tla.ptr<f32, gm, 4>>
      tla.copy %l0_a, %l1_a :
        !tla.tensor<!tla.layout<!tla.shape<32>, !tla.stride<1>, !tla.shape<32>, VectorLayout>, !tla.coord<0>, !tla.ptr<f32, l0a, 512>>,
        !tla.tensor<!tla.layout<!tla.shape<32>, !tla.stride<1>, !tla.shape<32>, VectorLayout>, !tla.coord<0>, !tla.ptr<f32, l1, 512>>
      tla.copy %l0_b, %l1_b :
        !tla.tensor<!tla.layout<!tla.shape<(8,4),(16,2)>, !tla.stride<(1,256),(8,128)>, !tla.shape<32,32>, nZ>, !tla.coord<0,0>, !tla.ptr<f32, l0b, 512>>,
        !tla.tensor<!tla.layout<!tla.shape<(8,4),(16,2)>, !tla.stride<(1,256),(8,128)>, !tla.shape<32,32>, nZ>, !tla.coord<0,0>, !tla.ptr<f32, l1, 512>>
      %init = arith.constant true
      %unit_flag = arith.constant 3 : i64
      tla.mmad %l0_c, %l0_a, %l0_b, %init, %unit_flag {compute_order = #tla.compute_order<M_FIRST>, hf32_mode = #tla.hf32_mode<HF32_DISABLE>} :
        !tla.tensor<!tla.layout<!tla.shape<(16,1),(16,2)>, !tla.stride<(16,256),(1,256)>, !tla.shape<1,32>, L0Clayout>, !tla.coord<0,0>, !tla.ptr<f32, l0c, 1024>>,
        !tla.tensor<!tla.layout<!tla.shape<32>, !tla.stride<1>, !tla.shape<32>, VectorLayout>, !tla.coord<0>, !tla.ptr<f32, l0a, 512>>,
        !tla.tensor<!tla.layout<!tla.shape<(8,4),(16,2)>, !tla.stride<(1,256),(8,128)>, !tla.shape<32,32>, nZ>, !tla.coord<0,0>, !tla.ptr<f32, l0b, 512>>,
        i1, i64
      %params = "tla.CopyL0C2DstParams"() <{unit_flag = 3 : i64, relu_enable = false, quant_mode = #tla.quant_mode<NO_QUANT>, l0c2ub_mode = #tla.l0c2ub_mode<NO_SPLIT_VEC_0>}> : () -> !tla.copy_l0c2dst_params
      "tla.copy"(%arg_c, %l0_c, %params) :
        (!tla.tensor<!tla.layout<!tla.shape<1,32>, !tla.stride<32,1>, !tla.shape<1,32>, RowMajor>, !tla.coord<0,0>, !tla.ptr<f32, gm, 4>>,
         !tla.tensor<!tla.layout<!tla.shape<(16,1),(16,2)>, !tla.stride<(16,256),(1,256)>, !tla.shape<1,32>, L0Clayout>, !tla.coord<0,0>, !tla.ptr<f32, l0c, 1024>>,
         !tla.copy_l0c2dst_params) -> ()
    }
    tla.return
  }
}

// The GEMV flag reaches the bc wrapper as its last argument.
// CHECK: func.func private @mmad_float_float_float({{.*}}, i1, i8, i1)
// CHECK: func.func private @copy_l0c_to_gm_RowMajor_float_float(
// The B block keeps its nZ fractal packing on both legs.
// CHECK: func.func private @copy_l1_nZ_to_l0b_nZ_float(
// The A row moves VectorLayout to VectorLayout on both legs; no zN route is
// taken, and no separate GEMV-only copy op exists.
// CHECK: func.func private @copy_l1_VectorLayout_to_l0a_VectorLayout_float(
// CHECK: func.func private @copy_gm_nZ_to_l1_nZ_float(
// CHECK: func.func private @copy_gm_VectorLayout_to_l1_VectorLayout_float(
// CHECK: func.func @gemv_mmad(
// CHECK: %[[GEMV:.*]] = llvm.mlir.constant(true) : i1
// CHECK: call @copy_gm_VectorLayout_to_l1_VectorLayout_float(
// CHECK: call @copy_gm_nZ_to_l1_nZ_float(
// CHECK: call @copy_l1_VectorLayout_to_l0a_VectorLayout_float(
// CHECK: call @copy_l1_nZ_to_l0b_nZ_float(
// CHECK: call @mmad_float_float_float({{.*}}%[[GEMV]]) :
// The accumulator leaves through the ordinary RowMajor fixpipe: the gemv result
// is written as a row, with no separate destination path.
// CHECK: call @copy_l0c_to_gm_RowMajor_float_float(
// CHECK-NOT: llvm.mlir.constant(false) : i1
