// RUN: %tla_compile %s -o - | %filecheck %s
//
// GM zN -> L1 zN: the GM tile is already packed in fractal order, so the copy
// is a plain block move -- no Nd2Nz reorder and no element-format conversion.
// The source is a prepacked GM tile reached through tensor_ptr + a zN
// make_tensor, which is how such a buffer is spelled.
//
// The u32 case pins the unsigned side of the same route: its meta-op symbol is
// only usable if the call carries the `_mlir_ciface_` wrapper and the AIC core
// type, so both attributes are checked here.
//
// Only the aligned zN is a GM source tag; the tight zNUnAlign packing belongs to
// the UB -> L1 route and is refused here.

module attributes {tla.module_exec_units = "cube"} {
  tla.func @copy_gm_zn_to_l1_zn(
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
    %shape = tla.make_shape -> !tla.shape<(16,2),(8,4)>
    %stride = tla.make_stride -> !tla.stride<(8,128),(1,256)>
    %origin = tla.make_shape -> !tla.shape<32,32>
    %layout = tla.make_layout %shape, %stride origin %origin {layoutTag = #tla.layout_tag<zN>} :
      !tla.shape<(16,2),(8,4)>, !tla.stride<(8,128),(1,256)> origin !tla.shape<32,32> ->
      !tla.layout<!tla.shape<(16,2),(8,4)>, !tla.stride<(8,128),(1,256)>, !tla.shape<32,32>, zN>
    %coord = tla.make_coord -> !tla.coord<0,0>
    %src = tla.make_tensor %src_ptr, %layout, %coord :
      !tla.ptr<f32, gm, 4>,
      !tla.layout<!tla.shape<(16,2),(8,4)>, !tla.stride<(8,128),(1,256)>, !tla.shape<32,32>, zN>,
      !tla.coord<0,0> ->
      !tla.tensor<!tla.layout<!tla.shape<(16,2),(8,4)>, !tla.stride<(8,128),(1,256)>, !tla.shape<32,32>, zN>, !tla.coord<0,0>, !tla.ptr<f32, gm, 4>>
    %dst = tla.make_tensor %l1ptr, %layout, %coord :
      !tla.ptr<f32, l1, 512>,
      !tla.layout<!tla.shape<(16,2),(8,4)>, !tla.stride<(8,128),(1,256)>, !tla.shape<32,32>, zN>,
      !tla.coord<0,0> ->
      !tla.tensor<!tla.layout<!tla.shape<(16,2),(8,4)>, !tla.stride<(8,128),(1,256)>, !tla.shape<32,32>, zN>, !tla.coord<0,0>, !tla.ptr<f32, l1, 512>>
    "tla.cube"() ({
      tla.copy %dst, %src :
        !tla.tensor<!tla.layout<!tla.shape<(16,2),(8,4)>, !tla.stride<(8,128),(1,256)>, !tla.shape<32,32>, zN>, !tla.coord<0,0>, !tla.ptr<f32, l1, 512>>,
        !tla.tensor<!tla.layout<!tla.shape<(16,2),(8,4)>, !tla.stride<(8,128),(1,256)>, !tla.shape<32,32>, zN>, !tla.coord<0,0>, !tla.ptr<f32, gm, 4>>
    }) : () -> ()
    tla.return
  }

  // Same route on an unsigned element type: u32 is registered for GM -> L1.
  tla.func @copy_gm_zn_to_l1_zn_u32(
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
    %shape = tla.make_shape -> !tla.shape<(16,2),(8,4)>
    %stride = tla.make_stride -> !tla.stride<(8,128),(1,256)>
    %origin = tla.make_shape -> !tla.shape<32,32>
    %layout = tla.make_layout %shape, %stride origin %origin {layoutTag = #tla.layout_tag<zN>} :
      !tla.shape<(16,2),(8,4)>, !tla.stride<(8,128),(1,256)> origin !tla.shape<32,32> ->
      !tla.layout<!tla.shape<(16,2),(8,4)>, !tla.stride<(8,128),(1,256)>, !tla.shape<32,32>, zN>
    %coord = tla.make_coord -> !tla.coord<0,0>
    %src = tla.make_tensor %src_ptr, %layout, %coord :
      !tla.ptr<ui32, gm, 4>,
      !tla.layout<!tla.shape<(16,2),(8,4)>, !tla.stride<(8,128),(1,256)>, !tla.shape<32,32>, zN>,
      !tla.coord<0,0> ->
      !tla.tensor<!tla.layout<!tla.shape<(16,2),(8,4)>, !tla.stride<(8,128),(1,256)>, !tla.shape<32,32>, zN>, !tla.coord<0,0>, !tla.ptr<ui32, gm, 4>>
    %dst = tla.make_tensor %l1ptr, %layout, %coord :
      !tla.ptr<ui32, l1, 512>,
      !tla.layout<!tla.shape<(16,2),(8,4)>, !tla.stride<(8,128),(1,256)>, !tla.shape<32,32>, zN>,
      !tla.coord<0,0> ->
      !tla.tensor<!tla.layout<!tla.shape<(16,2),(8,4)>, !tla.stride<(8,128),(1,256)>, !tla.shape<32,32>, zN>, !tla.coord<0,0>, !tla.ptr<ui32, l1, 512>>
    "tla.cube"() ({
      tla.copy %dst, %src :
        !tla.tensor<!tla.layout<!tla.shape<(16,2),(8,4)>, !tla.stride<(8,128),(1,256)>, !tla.shape<32,32>, zN>, !tla.coord<0,0>, !tla.ptr<ui32, l1, 512>>,
        !tla.tensor<!tla.layout<!tla.shape<(16,2),(8,4)>, !tla.stride<(8,128),(1,256)>, !tla.shape<32,32>, zN>, !tla.coord<0,0>, !tla.ptr<ui32, gm, 4>>
    }) : () -> ()
    tla.return
  }
}

// Both stubs are declared up front, with the wrappers the bc symbol names need.
// CHECK-DAG: func.func private @copy_gm_zN_to_l1_zN_float{{.*}}hivm.func_core_type = #hivm.func_core_type<AIC>{{.*}}llvm.emit_c_interface
// CHECK-DAG: func.func private @copy_gm_zN_to_l1_zN_uint32_t{{.*}}hivm.func_core_type = #hivm.func_core_type<AIC>{{.*}}llvm.emit_c_interface
// The GM source materializes as the rank-2 view its ciface stub takes, sized by
// the logical origin extents of the fractal tile. The rank is spelled out
// (two `?` dimensions): a rank-1 view would satisfy the stub's type only by
// accident and would not be castable to it.
// CHECK-LABEL: func.func @copy_gm_zn_to_l1_zn
// CHECK: hivm.hir.pointer_cast{{.*}} : memref<?x?xf32, #hivm.address_space<gm>>
// CHECK: call @copy_gm_zN_to_l1_zN_float
// CHECK: hivm.hir.pointer_cast{{.*}} : memref<?x?xui32, #hivm.address_space<gm>>
// CHECK: call @copy_gm_zN_to_l1_zN_uint32_t
// Only the aligned zN is a GM source tag: a zNUnAlign GM tile is refused by the
// route table (see tests/frontend/ops/copy/test_gm_zn_to_l1_zn_route.py) rather
// than lowered, so no second function is pinned here.
// CHECK-NOT: "tla.copy"
// CHECK-NOT: "tla.make_tensor"
