// RUN: %tla_compile %s -o - | %filecheck %s
//
// GM nZ -> L1 nZ: the transposed mirror of the zN block move. The GM tile is
// already packed as C0-row by 16-column fractals, so both endpoints carry the
// same packing and the copy reorders nothing. The source is reached through
// tensor_ptr + an nZ make_tensor, which is how a prepacked column-major weight
// buffer is spelled.
//
// The u32 case pins the unsigned side of the same route: its meta-op symbol is
// only usable if the call carries the `_mlir_ciface_` wrapper and the AIC core
// type, so both attributes are checked here.

module attributes {tla.module_exec_units = "cube"} {
  tla.func @copy_gm_nz_to_l1_nz(
      %arg0: !tla.tensor<!tla.layout<!tla.shape<32,32>, !tla.stride<32,1>, !tla.shape<32,32>, RowMajor>, !tla.coord<0,0>, !tla.ptr<f32, gm, 4>>) {
    %raw = tla.alloc_ptr{size_bytes = 4096} -> !tla.ptr<i8, l1, 512>
    %l1ptr = tla.recast_ptr %raw : !tla.ptr<i8, l1, 512> -> !tla.ptr<f32, l1, 512>
    %tile_shape = tla.make_shape -> !tla.shape<32,32>
    %tile_coord = tla.make_coord -> !tla.coord<0,0>
    %src_tile = tla.tile_view %arg0, %tile_shape, %tile_coord :
      !tla.tensor<!tla.layout<!tla.shape<32,32>, !tla.stride<32,1>, !tla.shape<32,32>, RowMajor>, !tla.coord<0,0>, !tla.ptr<f32, gm, 4>>,
      !tla.shape<32,32>, !tla.coord<0,0> ->
      !tla.tensor<!tla.layout<!tla.shape<32,32>, !tla.stride<32,1>, !tla.shape<32,32>, RowMajor>, !tla.coord<0,0>, !tla.ptr<f32, gm, 4>>
    %src_ptr = tla.tensor_ptr %src_tile :
      !tla.tensor<!tla.layout<!tla.shape<32,32>, !tla.stride<32,1>, !tla.shape<32,32>, RowMajor>, !tla.coord<0,0>, !tla.ptr<f32, gm, 4>> ->
      !tla.ptr<f32, gm, 4>
    %shape = tla.make_shape -> !tla.shape<(8,4),(16,2)>
    %stride = tla.make_stride -> !tla.stride<(1,256),(8,128)>
    %origin = tla.make_shape -> !tla.shape<32,32>
    %layout = tla.make_layout %shape, %stride origin %origin {layoutTag = #tla.layout_tag<nZ>} :
      !tla.shape<(8,4),(16,2)>, !tla.stride<(1,256),(8,128)> origin !tla.shape<32,32> ->
      !tla.layout<!tla.shape<(8,4),(16,2)>, !tla.stride<(1,256),(8,128)>, !tla.shape<32,32>, nZ>
    %coord = tla.make_coord -> !tla.coord<0,0>
    %src = tla.make_tensor %src_ptr, %layout, %coord :
      !tla.ptr<f32, gm, 4>,
      !tla.layout<!tla.shape<(8,4),(16,2)>, !tla.stride<(1,256),(8,128)>, !tla.shape<32,32>, nZ>,
      !tla.coord<0,0> ->
      !tla.tensor<!tla.layout<!tla.shape<(8,4),(16,2)>, !tla.stride<(1,256),(8,128)>, !tla.shape<32,32>, nZ>, !tla.coord<0,0>, !tla.ptr<f32, gm, 4>>
    %dst = tla.make_tensor %l1ptr, %layout, %coord :
      !tla.ptr<f32, l1, 512>,
      !tla.layout<!tla.shape<(8,4),(16,2)>, !tla.stride<(1,256),(8,128)>, !tla.shape<32,32>, nZ>,
      !tla.coord<0,0> ->
      !tla.tensor<!tla.layout<!tla.shape<(8,4),(16,2)>, !tla.stride<(1,256),(8,128)>, !tla.shape<32,32>, nZ>, !tla.coord<0,0>, !tla.ptr<f32, l1, 512>>
    "tla.cube"() ({
      tla.copy %dst, %src :
        !tla.tensor<!tla.layout<!tla.shape<(8,4),(16,2)>, !tla.stride<(1,256),(8,128)>, !tla.shape<32,32>, nZ>, !tla.coord<0,0>, !tla.ptr<f32, l1, 512>>,
        !tla.tensor<!tla.layout<!tla.shape<(8,4),(16,2)>, !tla.stride<(1,256),(8,128)>, !tla.shape<32,32>, nZ>, !tla.coord<0,0>, !tla.ptr<f32, gm, 4>>
    }) : () -> ()
    tla.return
  }

  // Same route on an unsigned element type: u32 is registered for GM -> L1.
  tla.func @copy_gm_nz_to_l1_nz_u32(
      %arg0: !tla.tensor<!tla.layout<!tla.shape<32,32>, !tla.stride<32,1>, !tla.shape<32,32>, RowMajor>, !tla.coord<0,0>, !tla.ptr<ui32, gm, 4>>) {
    %raw = tla.alloc_ptr{size_bytes = 4096} -> !tla.ptr<i8, l1, 512>
    %l1ptr = tla.recast_ptr %raw : !tla.ptr<i8, l1, 512> -> !tla.ptr<ui32, l1, 512>
    %tile_shape = tla.make_shape -> !tla.shape<32,32>
    %tile_coord = tla.make_coord -> !tla.coord<0,0>
    %src_tile = tla.tile_view %arg0, %tile_shape, %tile_coord :
      !tla.tensor<!tla.layout<!tla.shape<32,32>, !tla.stride<32,1>, !tla.shape<32,32>, RowMajor>, !tla.coord<0,0>, !tla.ptr<ui32, gm, 4>>,
      !tla.shape<32,32>, !tla.coord<0,0> ->
      !tla.tensor<!tla.layout<!tla.shape<32,32>, !tla.stride<32,1>, !tla.shape<32,32>, RowMajor>, !tla.coord<0,0>, !tla.ptr<ui32, gm, 4>>
    %src_ptr = tla.tensor_ptr %src_tile :
      !tla.tensor<!tla.layout<!tla.shape<32,32>, !tla.stride<32,1>, !tla.shape<32,32>, RowMajor>, !tla.coord<0,0>, !tla.ptr<ui32, gm, 4>> ->
      !tla.ptr<ui32, gm, 4>
    %shape = tla.make_shape -> !tla.shape<(8,4),(16,2)>
    %stride = tla.make_stride -> !tla.stride<(1,256),(8,128)>
    %origin = tla.make_shape -> !tla.shape<32,32>
    %layout = tla.make_layout %shape, %stride origin %origin {layoutTag = #tla.layout_tag<nZ>} :
      !tla.shape<(8,4),(16,2)>, !tla.stride<(1,256),(8,128)> origin !tla.shape<32,32> ->
      !tla.layout<!tla.shape<(8,4),(16,2)>, !tla.stride<(1,256),(8,128)>, !tla.shape<32,32>, nZ>
    %coord = tla.make_coord -> !tla.coord<0,0>
    %src = tla.make_tensor %src_ptr, %layout, %coord :
      !tla.ptr<ui32, gm, 4>,
      !tla.layout<!tla.shape<(8,4),(16,2)>, !tla.stride<(1,256),(8,128)>, !tla.shape<32,32>, nZ>,
      !tla.coord<0,0> ->
      !tla.tensor<!tla.layout<!tla.shape<(8,4),(16,2)>, !tla.stride<(1,256),(8,128)>, !tla.shape<32,32>, nZ>, !tla.coord<0,0>, !tla.ptr<ui32, gm, 4>>
    %dst = tla.make_tensor %l1ptr, %layout, %coord :
      !tla.ptr<ui32, l1, 512>,
      !tla.layout<!tla.shape<(8,4),(16,2)>, !tla.stride<(1,256),(8,128)>, !tla.shape<32,32>, nZ>,
      !tla.coord<0,0> ->
      !tla.tensor<!tla.layout<!tla.shape<(8,4),(16,2)>, !tla.stride<(1,256),(8,128)>, !tla.shape<32,32>, nZ>, !tla.coord<0,0>, !tla.ptr<ui32, l1, 512>>
    "tla.cube"() ({
      tla.copy %dst, %src :
        !tla.tensor<!tla.layout<!tla.shape<(8,4),(16,2)>, !tla.stride<(1,256),(8,128)>, !tla.shape<32,32>, nZ>, !tla.coord<0,0>, !tla.ptr<ui32, l1, 512>>,
        !tla.tensor<!tla.layout<!tla.shape<(8,4),(16,2)>, !tla.stride<(1,256),(8,128)>, !tla.shape<32,32>, nZ>, !tla.coord<0,0>, !tla.ptr<ui32, gm, 4>>
    }) : () -> ()
    tla.return
  }

  // N = 20: not a multiple of 16, so both sides pad it to 32 columns and the
  // pitch in the descriptor is 32 * C0 = 256, not 20 * C0. There is no unaligned
  // nZ source tag at all: the block move steps whole C0-row groups and whole
  // logical columns, so it needs the padding on both axes and a tight packing
  // cannot be expressed. M = 20 is likewise rounded up to 24 rows, and the last
  // row group travels in full -- which is why both buffers below are sized 24x32
  // and not 20x20.
  tla.func @copy_gm_nz_padded_to_l1_nz(
      %arg0: !tla.tensor<!tla.layout<!tla.shape<20,20>, !tla.stride<20,1>, !tla.shape<20,20>, RowMajor>, !tla.coord<0,0>, !tla.ptr<f32, gm, 4>>) {
    // The padded tile is 3 row groups by 2 column groups of 128 f32: 3072 bytes,
    // not 20 * 20 * 4 = 1600. A logical-sized buffer would be read and written
    // past its end.
    %raw = tla.alloc_ptr{size_bytes = 3072} -> !tla.ptr<i8, l1, 512>
    %l1ptr = tla.recast_ptr %raw : !tla.ptr<i8, l1, 512> -> !tla.ptr<f32, l1, 512>
    %tile_shape = tla.make_shape -> !tla.shape<20,20>
    %tile_coord = tla.make_coord -> !tla.coord<0,0>
    %tile = tla.tile_view %arg0, %tile_shape, %tile_coord :
      !tla.tensor<!tla.layout<!tla.shape<20,20>, !tla.stride<20,1>, !tla.shape<20,20>, RowMajor>, !tla.coord<0,0>, !tla.ptr<f32, gm, 4>>,
      !tla.shape<20,20>, !tla.coord<0,0> ->
      !tla.tensor<!tla.layout<!tla.shape<20,20>, !tla.stride<20,1>, !tla.shape<20,20>, RowMajor>, !tla.coord<0,0>, !tla.ptr<f32, gm, 4>>
    %src_ptr = tla.tensor_ptr %tile :
      !tla.tensor<!tla.layout<!tla.shape<20,20>, !tla.stride<20,1>, !tla.shape<20,20>, RowMajor>, !tla.coord<0,0>, !tla.ptr<f32, gm, 4>> ->
      !tla.ptr<f32, gm, 4>
    %gs = tla.make_shape -> !tla.shape<(8,3),(16,2)>
    %gd = tla.make_stride -> !tla.stride<(1,256),(8,128)>
    %go = tla.make_shape -> !tla.shape<20,20>
    %gl = tla.make_layout %gs, %gd origin %go {layoutTag = #tla.layout_tag<nZ>} :
      !tla.shape<(8,3),(16,2)>, !tla.stride<(1,256),(8,128)> origin !tla.shape<20,20> ->
      !tla.layout<!tla.shape<(8,3),(16,2)>, !tla.stride<(1,256),(8,128)>, !tla.shape<20,20>, nZ>
    %gc = tla.make_coord -> !tla.coord<0,0>
    %src = tla.make_tensor %src_ptr, %gl, %gc :
      !tla.ptr<f32, gm, 4>,
      !tla.layout<!tla.shape<(8,3),(16,2)>, !tla.stride<(1,256),(8,128)>, !tla.shape<20,20>, nZ>,
      !tla.coord<0,0> ->
      !tla.tensor<!tla.layout<!tla.shape<(8,3),(16,2)>, !tla.stride<(1,256),(8,128)>, !tla.shape<20,20>, nZ>, !tla.coord<0,0>, !tla.ptr<f32, gm, 4>>
    %dst = tla.make_tensor_like %l1ptr like %src layoutTag(#tla.layout_tag<nZ>) :
      !tla.ptr<f32, l1, 512>,
      !tla.tensor<!tla.layout<!tla.shape<(8,3),(16,2)>, !tla.stride<(1,256),(8,128)>, !tla.shape<20,20>, nZ>, !tla.coord<0,0>, !tla.ptr<f32, gm, 4>> ->
      !tla.tensor<!tla.layout<!tla.shape<(8,3),(16,2)>, !tla.stride<(1,256),(8,128)>, !tla.shape<20,20>, nZ>, !tla.coord<0,0>, !tla.ptr<f32, l1, 512>>
    "tla.cube"() ({
      tla.copy %dst, %src :
        !tla.tensor<!tla.layout<!tla.shape<(8,3),(16,2)>, !tla.stride<(1,256),(8,128)>, !tla.shape<20,20>, nZ>, !tla.coord<0,0>, !tla.ptr<f32, l1, 512>>,
        !tla.tensor<!tla.layout<!tla.shape<(8,3),(16,2)>, !tla.stride<(1,256),(8,128)>, !tla.shape<20,20>, nZ>, !tla.coord<0,0>, !tla.ptr<f32, gm, 4>>
    }) : () -> ()
    tla.return
  }
}

// Both stubs are declared up front, with the wrappers the bc symbol names need.
// CHECK-DAG: func.func private @copy_gm_nZ_to_l1_nZ_float{{.*}}hivm.func_core_type = #hivm.func_core_type<AIC>{{.*}}llvm.emit_c_interface
// CHECK-DAG: func.func private @copy_gm_nZ_to_l1_nZ_uint32_t{{.*}}hivm.func_core_type = #hivm.func_core_type<AIC>{{.*}}llvm.emit_c_interface
// The GM source materializes as the rank-2 view its ciface stub takes, sized by
// the logical origin extents of the fractal tile. The rank is spelled out
// (two `?` dimensions): a rank-1 view would satisfy the stub's type only by
// accident and would not be castable to it.
// CHECK-LABEL: func.func @copy_gm_nz_to_l1_nz
// CHECK: hivm.hir.pointer_cast{{.*}} : memref<?x?xf32, #hivm.address_space<gm>>
// CHECK: call @copy_gm_nZ_to_l1_nZ_float
// CHECK: hivm.hir.pointer_cast{{.*}} : memref<?x?xui32, #hivm.address_space<gm>>
// CHECK: call @copy_gm_nZ_to_l1_nZ_uint32_t
// The padded tile reaches the same symbol; only the descriptor differs (the
// pitch stays 256 because both sides round N up to 16 columns).
// CHECK-LABEL: func.func @copy_gm_nz_padded_to_l1_nz
// CHECK: call @copy_gm_nZ_to_l1_nZ_float
// CHECK-NOT: "tla.copy"
// CHECK-NOT: "tla.make_tensor"
