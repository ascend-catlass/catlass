// RUN: %tla_compile %s -o - | %filecheck %s
//
// GM MxScale -> L1 MxScale: the scale block reaches L1 with its fractal reorder
// done by the DMA, and the GM side is a ptr-backed fractal tile (the tag is
// NZFamily, so it has no 2-D physical shape). Its ciface stub takes a rank-2
// `memref_t<__gm__ uint8_t, 2>*`, so the materialized view must be rank 2 --
// a rank-1 view would describe a different memref descriptor at the C ABI.

module attributes {tla.module_exec_units = "cube"} {
  tla.func @copy_gm_mxscale_a_to_l1(
      %arg0: !tla.tensor<!tla.layout<!tla.shape<256,16>, !tla.stride<16,1>, !tla.shape<256,16>, RowMajor>, !tla.coord<0,0>, !tla.ptr<!tla.f8e8m0, gm, 4>>) {
    %raw = tla.alloc_ptr{size_bytes = 4096} -> !tla.ptr<i8, l1, 512>
    %l1ptr = tla.recast_ptr %raw : !tla.ptr<i8, l1, 512> -> !tla.ptr<!tla.f8e8m0, l1, 512>
    %tile_shape = tla.make_shape -> !tla.shape<256,16>
    %tile_coord = tla.make_coord -> !tla.coord<0,0>
    %tile = tla.tile_view %arg0, %tile_shape, %tile_coord :
      !tla.tensor<!tla.layout<!tla.shape<256,16>, !tla.stride<16,1>, !tla.shape<256,16>, RowMajor>, !tla.coord<0,0>, !tla.ptr<!tla.f8e8m0, gm, 4>>,
      !tla.shape<256,16>, !tla.coord<0,0> ->
      !tla.tensor<!tla.layout<!tla.shape<256,16>, !tla.stride<16,1>, !tla.shape<256,16>, RowMajor>, !tla.coord<0,0>, !tla.ptr<!tla.f8e8m0, gm, 4>>
    %src_ptr = tla.tensor_ptr %tile :
      !tla.tensor<!tla.layout<!tla.shape<256,16>, !tla.stride<16,1>, !tla.shape<256,16>, RowMajor>, !tla.coord<0,0>, !tla.ptr<!tla.f8e8m0, gm, 4>> ->
      !tla.ptr<!tla.f8e8m0, gm, 4>
    %gs = tla.make_shape -> !tla.shape<(1,256),(2,8)>
    %gd = tla.make_stride -> !tla.stride<(0,16),(1,2)>
    %go = tla.make_shape -> !tla.shape<256,16>
    %gl = tla.make_layout %gs, %gd origin %go {layoutTag = #tla.layout_tag<RowMajorMxScaleA>} :
      !tla.shape<(1,256),(2,8)>, !tla.stride<(0,16),(1,2)> origin !tla.shape<256,16> ->
      !tla.layout<!tla.shape<(1,256),(2,8)>, !tla.stride<(0,16),(1,2)>, !tla.shape<256,16>, RowMajorMxScaleA>
    %gc = tla.make_coord -> !tla.coord<0,0>
    %gm = tla.make_tensor %src_ptr, %gl, %gc :
      !tla.ptr<!tla.f8e8m0, gm, 4>,
      !tla.layout<!tla.shape<(1,256),(2,8)>, !tla.stride<(0,16),(1,2)>, !tla.shape<256,16>, RowMajorMxScaleA>,
      !tla.coord<0,0> ->
      !tla.tensor<!tla.layout<!tla.shape<(1,256),(2,8)>, !tla.stride<(0,16),(1,2)>, !tla.shape<256,16>, RowMajorMxScaleA>, !tla.coord<0,0>, !tla.ptr<!tla.f8e8m0, gm, 4>>
    %dst = tla.make_tensor_like %l1ptr like %gm layoutTag(#tla.layout_tag<zZMxScale>) :
      !tla.ptr<!tla.f8e8m0, l1, 512>,
      !tla.tensor<!tla.layout<!tla.shape<(1,256),(2,8)>, !tla.stride<(0,16),(1,2)>, !tla.shape<256,16>, RowMajorMxScaleA>, !tla.coord<0,0>, !tla.ptr<!tla.f8e8m0, gm, 4>> ->
      !tla.tensor<!tla.layout<!tla.shape<(16,16),(2,8)>, !tla.stride<(2,512),(1,32)>, !tla.shape<256,16>, zZMxScale>, !tla.coord<0,0>, !tla.ptr<!tla.f8e8m0, l1, 512>>
    "tla.cube"() ({
      tla.copy %dst, %gm :
        !tla.tensor<!tla.layout<!tla.shape<(16,16),(2,8)>, !tla.stride<(2,512),(1,32)>, !tla.shape<256,16>, zZMxScale>, !tla.coord<0,0>, !tla.ptr<!tla.f8e8m0, l1, 512>>,
        !tla.tensor<!tla.layout<!tla.shape<(1,256),(2,8)>, !tla.stride<(0,16),(1,2)>, !tla.shape<256,16>, RowMajorMxScaleA>, !tla.coord<0,0>, !tla.ptr<!tla.f8e8m0, gm, 4>>
    }) : () -> ()
    tla.return
  }
}

// CHECK: func.func private @copy_gm_RowMajorMxScaleA_to_l1_zZMxScale_uint8_t
// CHECK-SAME: hivm.func_core_type = #hivm.func_core_type<AIC>
// CHECK-LABEL: func.func @copy_gm_mxscale_a_to_l1
// CHECK: hivm.hir.pointer_cast{{.*}} : memref<?x?xi8, #hivm.address_space<gm>>
// CHECK: call @copy_gm_RowMajorMxScaleA_to_l1_zZMxScale_uint8_t
// CHECK-NOT: "tla.copy"
