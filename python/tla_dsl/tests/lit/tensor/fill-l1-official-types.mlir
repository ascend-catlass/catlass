// RUN: %tla_compile %s -o - | %filecheck %s
module {
  tla.func @official_zN_int16_t() {
    %raw = tla.alloc_ptr{size_bytes = 4096} -> !tla.ptr<i8, l1, 512>
    %ptr = tla.recast_ptr %raw : !tla.ptr<i8, l1, 512> -> !tla.ptr<i16, l1, 512>
    %shape = tla.make_shape -> !tla.shape<(16,2),(16,4)>
    %stride = tla.make_stride -> !tla.stride<(16,256),(1,512)>
    %origin = tla.make_shape -> !tla.shape<32,64>
    %layout = tla.make_layout %shape, %stride origin %origin {layoutTag = #tla.layout_tag<zN>} :
      !tla.shape<(16,2),(16,4)>, !tla.stride<(16,256),(1,512)> origin !tla.shape<32,64> ->
      !tla.layout<!tla.shape<(16,2),(16,4)>, !tla.stride<(16,256),(1,512)>, !tla.shape<32,64>, zN>
    %zero_coord = tla.make_coord -> !tla.coord<0,0>
    %dst = tla.make_tensor %ptr, %layout, %zero_coord :
      !tla.ptr<i16, l1, 512>,
      !tla.layout<!tla.shape<(16,2),(16,4)>, !tla.stride<(16,256),(1,512)>, !tla.shape<32,64>, zN>,
      !tla.coord<0,0> ->
      !tla.tensor<!tla.layout<!tla.shape<(16,2),(16,4)>, !tla.stride<(16,256),(1,512)>, !tla.shape<32,64>, zN>, !tla.coord<0,0>, !tla.ptr<i16, l1, 512>>
    %zero = arith.constant 65535 : i32
    %fill_coord = tla.make_coord -> !tla.coord<0,32>
    %fill_shape = tla.make_shape -> !tla.shape<32,32>
    "tla.cube"() ({
      tla.fill %dst, %zero, %fill_coord, %fill_shape :
        !tla.tensor<!tla.layout<!tla.shape<(16,2),(16,4)>, !tla.stride<(16,256),(1,512)>, !tla.shape<32,64>, zN>, !tla.coord<0,0>, !tla.ptr<i16, l1, 512>>,
        i32, !tla.coord<0,32>, !tla.shape<32,32>
    }) : () -> ()
    tla.return
  }

  tla.func @official_zN_uint16_t() {
    %raw = tla.alloc_ptr{size_bytes = 4096} -> !tla.ptr<i8, l1, 512>
    %ptr = tla.recast_ptr %raw : !tla.ptr<i8, l1, 512> -> !tla.ptr<ui16, l1, 512>
    %shape = tla.make_shape -> !tla.shape<(16,2),(16,4)>
    %stride = tla.make_stride -> !tla.stride<(16,256),(1,512)>
    %origin = tla.make_shape -> !tla.shape<32,64>
    %layout = tla.make_layout %shape, %stride origin %origin {layoutTag = #tla.layout_tag<zN>} :
      !tla.shape<(16,2),(16,4)>, !tla.stride<(16,256),(1,512)> origin !tla.shape<32,64> ->
      !tla.layout<!tla.shape<(16,2),(16,4)>, !tla.stride<(16,256),(1,512)>, !tla.shape<32,64>, zN>
    %zero_coord = tla.make_coord -> !tla.coord<0,0>
    %dst = tla.make_tensor %ptr, %layout, %zero_coord :
      !tla.ptr<ui16, l1, 512>,
      !tla.layout<!tla.shape<(16,2),(16,4)>, !tla.stride<(16,256),(1,512)>, !tla.shape<32,64>, zN>,
      !tla.coord<0,0> ->
      !tla.tensor<!tla.layout<!tla.shape<(16,2),(16,4)>, !tla.stride<(16,256),(1,512)>, !tla.shape<32,64>, zN>, !tla.coord<0,0>, !tla.ptr<ui16, l1, 512>>
    %zero = arith.constant 65535 : i32
    %fill_coord = tla.make_coord -> !tla.coord<0,32>
    %fill_shape = tla.make_shape -> !tla.shape<32,32>
    "tla.cube"() ({
      tla.fill %dst, %zero, %fill_coord, %fill_shape :
        !tla.tensor<!tla.layout<!tla.shape<(16,2),(16,4)>, !tla.stride<(16,256),(1,512)>, !tla.shape<32,64>, zN>, !tla.coord<0,0>, !tla.ptr<ui16, l1, 512>>,
        i32, !tla.coord<0,32>, !tla.shape<32,32>
    }) : () -> ()
    tla.return
  }

  tla.func @official_zN_half() {
    %raw = tla.alloc_ptr{size_bytes = 4096} -> !tla.ptr<i8, l1, 512>
    %ptr = tla.recast_ptr %raw : !tla.ptr<i8, l1, 512> -> !tla.ptr<f16, l1, 512>
    %shape = tla.make_shape -> !tla.shape<(16,2),(16,4)>
    %stride = tla.make_stride -> !tla.stride<(16,256),(1,512)>
    %origin = tla.make_shape -> !tla.shape<32,64>
    %layout = tla.make_layout %shape, %stride origin %origin {layoutTag = #tla.layout_tag<zN>} :
      !tla.shape<(16,2),(16,4)>, !tla.stride<(16,256),(1,512)> origin !tla.shape<32,64> ->
      !tla.layout<!tla.shape<(16,2),(16,4)>, !tla.stride<(16,256),(1,512)>, !tla.shape<32,64>, zN>
    %zero_coord = tla.make_coord -> !tla.coord<0,0>
    %dst = tla.make_tensor %ptr, %layout, %zero_coord :
      !tla.ptr<f16, l1, 512>,
      !tla.layout<!tla.shape<(16,2),(16,4)>, !tla.stride<(16,256),(1,512)>, !tla.shape<32,64>, zN>,
      !tla.coord<0,0> ->
      !tla.tensor<!tla.layout<!tla.shape<(16,2),(16,4)>, !tla.stride<(16,256),(1,512)>, !tla.shape<32,64>, zN>, !tla.coord<0,0>, !tla.ptr<f16, l1, 512>>
    %zero = arith.constant 15872 : i32
    %fill_coord = tla.make_coord -> !tla.coord<0,32>
    %fill_shape = tla.make_shape -> !tla.shape<32,32>
    "tla.cube"() ({
      tla.fill %dst, %zero, %fill_coord, %fill_shape :
        !tla.tensor<!tla.layout<!tla.shape<(16,2),(16,4)>, !tla.stride<(16,256),(1,512)>, !tla.shape<32,64>, zN>, !tla.coord<0,0>, !tla.ptr<f16, l1, 512>>,
        i32, !tla.coord<0,32>, !tla.shape<32,32>
    }) : () -> ()
    tla.return
  }

  tla.func @official_zN_bf16() {
    %raw = tla.alloc_ptr{size_bytes = 4096} -> !tla.ptr<i8, l1, 512>
    %ptr = tla.recast_ptr %raw : !tla.ptr<i8, l1, 512> -> !tla.ptr<bf16, l1, 512>
    %shape = tla.make_shape -> !tla.shape<(16,2),(16,4)>
    %stride = tla.make_stride -> !tla.stride<(16,256),(1,512)>
    %origin = tla.make_shape -> !tla.shape<32,64>
    %layout = tla.make_layout %shape, %stride origin %origin {layoutTag = #tla.layout_tag<zN>} :
      !tla.shape<(16,2),(16,4)>, !tla.stride<(16,256),(1,512)> origin !tla.shape<32,64> ->
      !tla.layout<!tla.shape<(16,2),(16,4)>, !tla.stride<(16,256),(1,512)>, !tla.shape<32,64>, zN>
    %zero_coord = tla.make_coord -> !tla.coord<0,0>
    %dst = tla.make_tensor %ptr, %layout, %zero_coord :
      !tla.ptr<bf16, l1, 512>,
      !tla.layout<!tla.shape<(16,2),(16,4)>, !tla.stride<(16,256),(1,512)>, !tla.shape<32,64>, zN>,
      !tla.coord<0,0> ->
      !tla.tensor<!tla.layout<!tla.shape<(16,2),(16,4)>, !tla.stride<(16,256),(1,512)>, !tla.shape<32,64>, zN>, !tla.coord<0,0>, !tla.ptr<bf16, l1, 512>>
    %zero = arith.constant 16320 : i32
    %fill_coord = tla.make_coord -> !tla.coord<0,32>
    %fill_shape = tla.make_shape -> !tla.shape<32,32>
    "tla.cube"() ({
      tla.fill %dst, %zero, %fill_coord, %fill_shape :
        !tla.tensor<!tla.layout<!tla.shape<(16,2),(16,4)>, !tla.stride<(16,256),(1,512)>, !tla.shape<32,64>, zN>, !tla.coord<0,0>, !tla.ptr<bf16, l1, 512>>,
        i32, !tla.coord<0,32>, !tla.shape<32,32>
    }) : () -> ()
    tla.return
  }

  tla.func @official_zN_int32_t() {
    %raw = tla.alloc_ptr{size_bytes = 4096} -> !tla.ptr<i8, l1, 512>
    %ptr = tla.recast_ptr %raw : !tla.ptr<i8, l1, 512> -> !tla.ptr<i32, l1, 512>
    %shape = tla.make_shape -> !tla.shape<(16,2),(8,8)>
    %stride = tla.make_stride -> !tla.stride<(8,128),(1,256)>
    %origin = tla.make_shape -> !tla.shape<32,64>
    %layout = tla.make_layout %shape, %stride origin %origin {layoutTag = #tla.layout_tag<zN>} :
      !tla.shape<(16,2),(8,8)>, !tla.stride<(8,128),(1,256)> origin !tla.shape<32,64> ->
      !tla.layout<!tla.shape<(16,2),(8,8)>, !tla.stride<(8,128),(1,256)>, !tla.shape<32,64>, zN>
    %zero_coord = tla.make_coord -> !tla.coord<0,0>
    %dst = tla.make_tensor %ptr, %layout, %zero_coord :
      !tla.ptr<i32, l1, 512>,
      !tla.layout<!tla.shape<(16,2),(8,8)>, !tla.stride<(8,128),(1,256)>, !tla.shape<32,64>, zN>,
      !tla.coord<0,0> ->
      !tla.tensor<!tla.layout<!tla.shape<(16,2),(8,8)>, !tla.stride<(8,128),(1,256)>, !tla.shape<32,64>, zN>, !tla.coord<0,0>, !tla.ptr<i32, l1, 512>>
    %zero = arith.constant -1 : i32
    %fill_coord = tla.make_coord -> !tla.coord<0,32>
    %fill_shape = tla.make_shape -> !tla.shape<32,32>
    "tla.cube"() ({
      tla.fill %dst, %zero, %fill_coord, %fill_shape :
        !tla.tensor<!tla.layout<!tla.shape<(16,2),(8,8)>, !tla.stride<(8,128),(1,256)>, !tla.shape<32,64>, zN>, !tla.coord<0,0>, !tla.ptr<i32, l1, 512>>,
        i32, !tla.coord<0,32>, !tla.shape<32,32>
    }) : () -> ()
    tla.return
  }

  tla.func @official_zN_uint32_t() {
    %raw = tla.alloc_ptr{size_bytes = 4096} -> !tla.ptr<i8, l1, 512>
    %ptr = tla.recast_ptr %raw : !tla.ptr<i8, l1, 512> -> !tla.ptr<ui32, l1, 512>
    %shape = tla.make_shape -> !tla.shape<(16,2),(8,8)>
    %stride = tla.make_stride -> !tla.stride<(8,128),(1,256)>
    %origin = tla.make_shape -> !tla.shape<32,64>
    %layout = tla.make_layout %shape, %stride origin %origin {layoutTag = #tla.layout_tag<zN>} :
      !tla.shape<(16,2),(8,8)>, !tla.stride<(8,128),(1,256)> origin !tla.shape<32,64> ->
      !tla.layout<!tla.shape<(16,2),(8,8)>, !tla.stride<(8,128),(1,256)>, !tla.shape<32,64>, zN>
    %zero_coord = tla.make_coord -> !tla.coord<0,0>
    %dst = tla.make_tensor %ptr, %layout, %zero_coord :
      !tla.ptr<ui32, l1, 512>,
      !tla.layout<!tla.shape<(16,2),(8,8)>, !tla.stride<(8,128),(1,256)>, !tla.shape<32,64>, zN>,
      !tla.coord<0,0> ->
      !tla.tensor<!tla.layout<!tla.shape<(16,2),(8,8)>, !tla.stride<(8,128),(1,256)>, !tla.shape<32,64>, zN>, !tla.coord<0,0>, !tla.ptr<ui32, l1, 512>>
    %zero = arith.constant -2147483647 : i32
    %fill_coord = tla.make_coord -> !tla.coord<0,32>
    %fill_shape = tla.make_shape -> !tla.shape<32,32>
    "tla.cube"() ({
      tla.fill %dst, %zero, %fill_coord, %fill_shape :
        !tla.tensor<!tla.layout<!tla.shape<(16,2),(8,8)>, !tla.stride<(8,128),(1,256)>, !tla.shape<32,64>, zN>, !tla.coord<0,0>, !tla.ptr<ui32, l1, 512>>,
        i32, !tla.coord<0,32>, !tla.shape<32,32>
    }) : () -> ()
    tla.return
  }

  tla.func @official_zN_float() {
    %raw = tla.alloc_ptr{size_bytes = 4096} -> !tla.ptr<i8, l1, 512>
    %ptr = tla.recast_ptr %raw : !tla.ptr<i8, l1, 512> -> !tla.ptr<f32, l1, 512>
    %shape = tla.make_shape -> !tla.shape<(16,2),(8,8)>
    %stride = tla.make_stride -> !tla.stride<(8,128),(1,256)>
    %origin = tla.make_shape -> !tla.shape<32,64>
    %layout = tla.make_layout %shape, %stride origin %origin {layoutTag = #tla.layout_tag<zN>} :
      !tla.shape<(16,2),(8,8)>, !tla.stride<(8,128),(1,256)> origin !tla.shape<32,64> ->
      !tla.layout<!tla.shape<(16,2),(8,8)>, !tla.stride<(8,128),(1,256)>, !tla.shape<32,64>, zN>
    %zero_coord = tla.make_coord -> !tla.coord<0,0>
    %dst = tla.make_tensor %ptr, %layout, %zero_coord :
      !tla.ptr<f32, l1, 512>,
      !tla.layout<!tla.shape<(16,2),(8,8)>, !tla.stride<(8,128),(1,256)>, !tla.shape<32,64>, zN>,
      !tla.coord<0,0> ->
      !tla.tensor<!tla.layout<!tla.shape<(16,2),(8,8)>, !tla.stride<(8,128),(1,256)>, !tla.shape<32,64>, zN>, !tla.coord<0,0>, !tla.ptr<f32, l1, 512>>
    %zero = arith.constant -1082130432 : i32
    %fill_coord = tla.make_coord -> !tla.coord<0,32>
    %fill_shape = tla.make_shape -> !tla.shape<32,32>
    "tla.cube"() ({
      tla.fill %dst, %zero, %fill_coord, %fill_shape :
        !tla.tensor<!tla.layout<!tla.shape<(16,2),(8,8)>, !tla.stride<(8,128),(1,256)>, !tla.shape<32,64>, zN>, !tla.coord<0,0>, !tla.ptr<f32, l1, 512>>,
        i32, !tla.coord<0,32>, !tla.shape<32,32>
    }) : () -> ()
    tla.return
  }

  tla.func @official_nZ_int16_t() {
    %raw = tla.alloc_ptr{size_bytes = 4096} -> !tla.ptr<i8, l1, 512>
    %ptr = tla.recast_ptr %raw : !tla.ptr<i8, l1, 512> -> !tla.ptr<i16, l1, 512>
    %shape = tla.make_shape -> !tla.shape<(16,2),(16,4)>
    %stride = tla.make_stride -> !tla.stride<(1,1024),(16,256)>
    %origin = tla.make_shape -> !tla.shape<32,64>
    %layout = tla.make_layout %shape, %stride origin %origin {layoutTag = #tla.layout_tag<nZ>} :
      !tla.shape<(16,2),(16,4)>, !tla.stride<(1,1024),(16,256)> origin !tla.shape<32,64> ->
      !tla.layout<!tla.shape<(16,2),(16,4)>, !tla.stride<(1,1024),(16,256)>, !tla.shape<32,64>, nZ>
    %zero_coord = tla.make_coord -> !tla.coord<0,0>
    %dst = tla.make_tensor %ptr, %layout, %zero_coord :
      !tla.ptr<i16, l1, 512>,
      !tla.layout<!tla.shape<(16,2),(16,4)>, !tla.stride<(1,1024),(16,256)>, !tla.shape<32,64>, nZ>,
      !tla.coord<0,0> ->
      !tla.tensor<!tla.layout<!tla.shape<(16,2),(16,4)>, !tla.stride<(1,1024),(16,256)>, !tla.shape<32,64>, nZ>, !tla.coord<0,0>, !tla.ptr<i16, l1, 512>>
    %zero = arith.constant 65535 : i32
    %fill_coord = tla.make_coord -> !tla.coord<0,32>
    %fill_shape = tla.make_shape -> !tla.shape<32,32>
    "tla.cube"() ({
      tla.fill %dst, %zero, %fill_coord, %fill_shape :
        !tla.tensor<!tla.layout<!tla.shape<(16,2),(16,4)>, !tla.stride<(1,1024),(16,256)>, !tla.shape<32,64>, nZ>, !tla.coord<0,0>, !tla.ptr<i16, l1, 512>>,
        i32, !tla.coord<0,32>, !tla.shape<32,32>
    }) : () -> ()
    tla.return
  }

  tla.func @official_nZ_uint16_t() {
    %raw = tla.alloc_ptr{size_bytes = 4096} -> !tla.ptr<i8, l1, 512>
    %ptr = tla.recast_ptr %raw : !tla.ptr<i8, l1, 512> -> !tla.ptr<ui16, l1, 512>
    %shape = tla.make_shape -> !tla.shape<(16,2),(16,4)>
    %stride = tla.make_stride -> !tla.stride<(1,1024),(16,256)>
    %origin = tla.make_shape -> !tla.shape<32,64>
    %layout = tla.make_layout %shape, %stride origin %origin {layoutTag = #tla.layout_tag<nZ>} :
      !tla.shape<(16,2),(16,4)>, !tla.stride<(1,1024),(16,256)> origin !tla.shape<32,64> ->
      !tla.layout<!tla.shape<(16,2),(16,4)>, !tla.stride<(1,1024),(16,256)>, !tla.shape<32,64>, nZ>
    %zero_coord = tla.make_coord -> !tla.coord<0,0>
    %dst = tla.make_tensor %ptr, %layout, %zero_coord :
      !tla.ptr<ui16, l1, 512>,
      !tla.layout<!tla.shape<(16,2),(16,4)>, !tla.stride<(1,1024),(16,256)>, !tla.shape<32,64>, nZ>,
      !tla.coord<0,0> ->
      !tla.tensor<!tla.layout<!tla.shape<(16,2),(16,4)>, !tla.stride<(1,1024),(16,256)>, !tla.shape<32,64>, nZ>, !tla.coord<0,0>, !tla.ptr<ui16, l1, 512>>
    %zero = arith.constant 65535 : i32
    %fill_coord = tla.make_coord -> !tla.coord<0,32>
    %fill_shape = tla.make_shape -> !tla.shape<32,32>
    "tla.cube"() ({
      tla.fill %dst, %zero, %fill_coord, %fill_shape :
        !tla.tensor<!tla.layout<!tla.shape<(16,2),(16,4)>, !tla.stride<(1,1024),(16,256)>, !tla.shape<32,64>, nZ>, !tla.coord<0,0>, !tla.ptr<ui16, l1, 512>>,
        i32, !tla.coord<0,32>, !tla.shape<32,32>
    }) : () -> ()
    tla.return
  }

  tla.func @official_nZ_half() {
    %raw = tla.alloc_ptr{size_bytes = 4096} -> !tla.ptr<i8, l1, 512>
    %ptr = tla.recast_ptr %raw : !tla.ptr<i8, l1, 512> -> !tla.ptr<f16, l1, 512>
    %shape = tla.make_shape -> !tla.shape<(16,2),(16,4)>
    %stride = tla.make_stride -> !tla.stride<(1,1024),(16,256)>
    %origin = tla.make_shape -> !tla.shape<32,64>
    %layout = tla.make_layout %shape, %stride origin %origin {layoutTag = #tla.layout_tag<nZ>} :
      !tla.shape<(16,2),(16,4)>, !tla.stride<(1,1024),(16,256)> origin !tla.shape<32,64> ->
      !tla.layout<!tla.shape<(16,2),(16,4)>, !tla.stride<(1,1024),(16,256)>, !tla.shape<32,64>, nZ>
    %zero_coord = tla.make_coord -> !tla.coord<0,0>
    %dst = tla.make_tensor %ptr, %layout, %zero_coord :
      !tla.ptr<f16, l1, 512>,
      !tla.layout<!tla.shape<(16,2),(16,4)>, !tla.stride<(1,1024),(16,256)>, !tla.shape<32,64>, nZ>,
      !tla.coord<0,0> ->
      !tla.tensor<!tla.layout<!tla.shape<(16,2),(16,4)>, !tla.stride<(1,1024),(16,256)>, !tla.shape<32,64>, nZ>, !tla.coord<0,0>, !tla.ptr<f16, l1, 512>>
    %zero = arith.constant 15872 : i32
    %fill_coord = tla.make_coord -> !tla.coord<0,32>
    %fill_shape = tla.make_shape -> !tla.shape<32,32>
    "tla.cube"() ({
      tla.fill %dst, %zero, %fill_coord, %fill_shape :
        !tla.tensor<!tla.layout<!tla.shape<(16,2),(16,4)>, !tla.stride<(1,1024),(16,256)>, !tla.shape<32,64>, nZ>, !tla.coord<0,0>, !tla.ptr<f16, l1, 512>>,
        i32, !tla.coord<0,32>, !tla.shape<32,32>
    }) : () -> ()
    tla.return
  }

  tla.func @official_nZ_bf16() {
    %raw = tla.alloc_ptr{size_bytes = 4096} -> !tla.ptr<i8, l1, 512>
    %ptr = tla.recast_ptr %raw : !tla.ptr<i8, l1, 512> -> !tla.ptr<bf16, l1, 512>
    %shape = tla.make_shape -> !tla.shape<(16,2),(16,4)>
    %stride = tla.make_stride -> !tla.stride<(1,1024),(16,256)>
    %origin = tla.make_shape -> !tla.shape<32,64>
    %layout = tla.make_layout %shape, %stride origin %origin {layoutTag = #tla.layout_tag<nZ>} :
      !tla.shape<(16,2),(16,4)>, !tla.stride<(1,1024),(16,256)> origin !tla.shape<32,64> ->
      !tla.layout<!tla.shape<(16,2),(16,4)>, !tla.stride<(1,1024),(16,256)>, !tla.shape<32,64>, nZ>
    %zero_coord = tla.make_coord -> !tla.coord<0,0>
    %dst = tla.make_tensor %ptr, %layout, %zero_coord :
      !tla.ptr<bf16, l1, 512>,
      !tla.layout<!tla.shape<(16,2),(16,4)>, !tla.stride<(1,1024),(16,256)>, !tla.shape<32,64>, nZ>,
      !tla.coord<0,0> ->
      !tla.tensor<!tla.layout<!tla.shape<(16,2),(16,4)>, !tla.stride<(1,1024),(16,256)>, !tla.shape<32,64>, nZ>, !tla.coord<0,0>, !tla.ptr<bf16, l1, 512>>
    %zero = arith.constant 16320 : i32
    %fill_coord = tla.make_coord -> !tla.coord<0,32>
    %fill_shape = tla.make_shape -> !tla.shape<32,32>
    "tla.cube"() ({
      tla.fill %dst, %zero, %fill_coord, %fill_shape :
        !tla.tensor<!tla.layout<!tla.shape<(16,2),(16,4)>, !tla.stride<(1,1024),(16,256)>, !tla.shape<32,64>, nZ>, !tla.coord<0,0>, !tla.ptr<bf16, l1, 512>>,
        i32, !tla.coord<0,32>, !tla.shape<32,32>
    }) : () -> ()
    tla.return
  }

  tla.func @official_nZ_int32_t() {
    %raw = tla.alloc_ptr{size_bytes = 4096} -> !tla.ptr<i8, l1, 512>
    %ptr = tla.recast_ptr %raw : !tla.ptr<i8, l1, 512> -> !tla.ptr<i32, l1, 512>
    %shape = tla.make_shape -> !tla.shape<(8,4),(16,4)>
    %stride = tla.make_stride -> !tla.stride<(1,512),(8,128)>
    %origin = tla.make_shape -> !tla.shape<32,64>
    %layout = tla.make_layout %shape, %stride origin %origin {layoutTag = #tla.layout_tag<nZ>} :
      !tla.shape<(8,4),(16,4)>, !tla.stride<(1,512),(8,128)> origin !tla.shape<32,64> ->
      !tla.layout<!tla.shape<(8,4),(16,4)>, !tla.stride<(1,512),(8,128)>, !tla.shape<32,64>, nZ>
    %zero_coord = tla.make_coord -> !tla.coord<0,0>
    %dst = tla.make_tensor %ptr, %layout, %zero_coord :
      !tla.ptr<i32, l1, 512>,
      !tla.layout<!tla.shape<(8,4),(16,4)>, !tla.stride<(1,512),(8,128)>, !tla.shape<32,64>, nZ>,
      !tla.coord<0,0> ->
      !tla.tensor<!tla.layout<!tla.shape<(8,4),(16,4)>, !tla.stride<(1,512),(8,128)>, !tla.shape<32,64>, nZ>, !tla.coord<0,0>, !tla.ptr<i32, l1, 512>>
    %zero = arith.constant -1 : i32
    %fill_coord = tla.make_coord -> !tla.coord<0,32>
    %fill_shape = tla.make_shape -> !tla.shape<32,32>
    "tla.cube"() ({
      tla.fill %dst, %zero, %fill_coord, %fill_shape :
        !tla.tensor<!tla.layout<!tla.shape<(8,4),(16,4)>, !tla.stride<(1,512),(8,128)>, !tla.shape<32,64>, nZ>, !tla.coord<0,0>, !tla.ptr<i32, l1, 512>>,
        i32, !tla.coord<0,32>, !tla.shape<32,32>
    }) : () -> ()
    tla.return
  }

  tla.func @official_nZ_uint32_t() {
    %raw = tla.alloc_ptr{size_bytes = 4096} -> !tla.ptr<i8, l1, 512>
    %ptr = tla.recast_ptr %raw : !tla.ptr<i8, l1, 512> -> !tla.ptr<ui32, l1, 512>
    %shape = tla.make_shape -> !tla.shape<(8,4),(16,4)>
    %stride = tla.make_stride -> !tla.stride<(1,512),(8,128)>
    %origin = tla.make_shape -> !tla.shape<32,64>
    %layout = tla.make_layout %shape, %stride origin %origin {layoutTag = #tla.layout_tag<nZ>} :
      !tla.shape<(8,4),(16,4)>, !tla.stride<(1,512),(8,128)> origin !tla.shape<32,64> ->
      !tla.layout<!tla.shape<(8,4),(16,4)>, !tla.stride<(1,512),(8,128)>, !tla.shape<32,64>, nZ>
    %zero_coord = tla.make_coord -> !tla.coord<0,0>
    %dst = tla.make_tensor %ptr, %layout, %zero_coord :
      !tla.ptr<ui32, l1, 512>,
      !tla.layout<!tla.shape<(8,4),(16,4)>, !tla.stride<(1,512),(8,128)>, !tla.shape<32,64>, nZ>,
      !tla.coord<0,0> ->
      !tla.tensor<!tla.layout<!tla.shape<(8,4),(16,4)>, !tla.stride<(1,512),(8,128)>, !tla.shape<32,64>, nZ>, !tla.coord<0,0>, !tla.ptr<ui32, l1, 512>>
    %zero = arith.constant -2147483647 : i32
    %fill_coord = tla.make_coord -> !tla.coord<0,32>
    %fill_shape = tla.make_shape -> !tla.shape<32,32>
    "tla.cube"() ({
      tla.fill %dst, %zero, %fill_coord, %fill_shape :
        !tla.tensor<!tla.layout<!tla.shape<(8,4),(16,4)>, !tla.stride<(1,512),(8,128)>, !tla.shape<32,64>, nZ>, !tla.coord<0,0>, !tla.ptr<ui32, l1, 512>>,
        i32, !tla.coord<0,32>, !tla.shape<32,32>
    }) : () -> ()
    tla.return
  }

  tla.func @official_nZ_float() {
    %raw = tla.alloc_ptr{size_bytes = 4096} -> !tla.ptr<i8, l1, 512>
    %ptr = tla.recast_ptr %raw : !tla.ptr<i8, l1, 512> -> !tla.ptr<f32, l1, 512>
    %shape = tla.make_shape -> !tla.shape<(8,4),(16,4)>
    %stride = tla.make_stride -> !tla.stride<(1,512),(8,128)>
    %origin = tla.make_shape -> !tla.shape<32,64>
    %layout = tla.make_layout %shape, %stride origin %origin {layoutTag = #tla.layout_tag<nZ>} :
      !tla.shape<(8,4),(16,4)>, !tla.stride<(1,512),(8,128)> origin !tla.shape<32,64> ->
      !tla.layout<!tla.shape<(8,4),(16,4)>, !tla.stride<(1,512),(8,128)>, !tla.shape<32,64>, nZ>
    %zero_coord = tla.make_coord -> !tla.coord<0,0>
    %dst = tla.make_tensor %ptr, %layout, %zero_coord :
      !tla.ptr<f32, l1, 512>,
      !tla.layout<!tla.shape<(8,4),(16,4)>, !tla.stride<(1,512),(8,128)>, !tla.shape<32,64>, nZ>,
      !tla.coord<0,0> ->
      !tla.tensor<!tla.layout<!tla.shape<(8,4),(16,4)>, !tla.stride<(1,512),(8,128)>, !tla.shape<32,64>, nZ>, !tla.coord<0,0>, !tla.ptr<f32, l1, 512>>
    %zero = arith.constant -1082130432 : i32
    %fill_coord = tla.make_coord -> !tla.coord<0,32>
    %fill_shape = tla.make_shape -> !tla.shape<32,32>
    "tla.cube"() ({
      tla.fill %dst, %zero, %fill_coord, %fill_shape :
        !tla.tensor<!tla.layout<!tla.shape<(8,4),(16,4)>, !tla.stride<(1,512),(8,128)>, !tla.shape<32,64>, nZ>, !tla.coord<0,0>, !tla.ptr<f32, l1, 512>>,
        i32, !tla.coord<0,32>, !tla.shape<32,32>
    }) : () -> ()
    tla.return
  }
}


// CHECK-DAG: func.func private @fill_l1_zN_int16_t(memref<?xi16, strided<[?], offset: ?>, #hivm.address_space<cbuf>>, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i32) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIC>, llvm.emit_c_interface}
// CHECK-DAG: func.func private @fill_l1_zN_uint16_t(memref<?xui16, strided<[?], offset: ?>, #hivm.address_space<cbuf>>, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i32) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIC>, llvm.emit_c_interface}
// CHECK-DAG: func.func private @fill_l1_zN_half(memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<cbuf>>, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i32) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIC>, llvm.emit_c_interface}
// CHECK-DAG: func.func private @fill_l1_zN_bf16(memref<?xbf16, strided<[?], offset: ?>, #hivm.address_space<cbuf>>, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i32) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIC>, llvm.emit_c_interface}
// CHECK-DAG: func.func private @fill_l1_zN_int32_t(memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<cbuf>>, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i32) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIC>, llvm.emit_c_interface}
// CHECK-DAG: func.func private @fill_l1_zN_uint32_t(memref<?xui32, strided<[?], offset: ?>, #hivm.address_space<cbuf>>, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i32) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIC>, llvm.emit_c_interface}
// CHECK-DAG: func.func private @fill_l1_zN_float(memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<cbuf>>, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i32) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIC>, llvm.emit_c_interface}
// CHECK-DAG: func.func private @fill_l1_nZ_int16_t(memref<?xi16, strided<[?], offset: ?>, #hivm.address_space<cbuf>>, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i32) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIC>, llvm.emit_c_interface}
// CHECK-DAG: func.func private @fill_l1_nZ_uint16_t(memref<?xui16, strided<[?], offset: ?>, #hivm.address_space<cbuf>>, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i32) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIC>, llvm.emit_c_interface}
// CHECK-DAG: func.func private @fill_l1_nZ_half(memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<cbuf>>, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i32) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIC>, llvm.emit_c_interface}
// CHECK-DAG: func.func private @fill_l1_nZ_bf16(memref<?xbf16, strided<[?], offset: ?>, #hivm.address_space<cbuf>>, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i32) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIC>, llvm.emit_c_interface}
// CHECK-DAG: func.func private @fill_l1_nZ_int32_t(memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<cbuf>>, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i32) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIC>, llvm.emit_c_interface}
// CHECK-DAG: func.func private @fill_l1_nZ_uint32_t(memref<?xui32, strided<[?], offset: ?>, #hivm.address_space<cbuf>>, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i32) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIC>, llvm.emit_c_interface}
// CHECK-DAG: func.func private @fill_l1_nZ_float(memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<cbuf>>, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i32) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIC>, llvm.emit_c_interface}
// CHECK-LABEL: func.func @official_zN_int16_t()
// CHECK: %[[BITS:.+]] = llvm.mlir.constant(65535 : i32) : i32
// CHECK: call @fill_l1_zN_int16_t
// CHECK-SAME: %[[BITS]])
// CHECK-NOT: tla.fill
// CHECK-LABEL: func.func @official_zN_uint16_t()
// CHECK: %[[BITS:.+]] = llvm.mlir.constant(65535 : i32) : i32
// CHECK: call @fill_l1_zN_uint16_t
// CHECK-SAME: %[[BITS]])
// CHECK-NOT: tla.fill
// CHECK-LABEL: func.func @official_zN_half()
// CHECK: %[[BITS:.+]] = llvm.mlir.constant(15872 : i32) : i32
// CHECK: call @fill_l1_zN_half
// CHECK-SAME: %[[BITS]])
// CHECK-NOT: tla.fill
// CHECK-LABEL: func.func @official_zN_bf16()
// CHECK: %[[BITS:.+]] = llvm.mlir.constant(16320 : i32) : i32
// CHECK: call @fill_l1_zN_bf16
// CHECK-SAME: %[[BITS]])
// CHECK-NOT: tla.fill
// CHECK-LABEL: func.func @official_zN_int32_t()
// CHECK: %[[BITS:.+]] = llvm.mlir.constant(-1 : i32) : i32
// CHECK: call @fill_l1_zN_int32_t
// CHECK-SAME: %[[BITS]])
// CHECK-NOT: tla.fill
// CHECK-LABEL: func.func @official_zN_uint32_t()
// CHECK: %[[BITS:.+]] = llvm.mlir.constant(-2147483647 : i32) : i32
// CHECK: call @fill_l1_zN_uint32_t
// CHECK-SAME: %[[BITS]])
// CHECK-NOT: tla.fill
// CHECK-LABEL: func.func @official_zN_float()
// CHECK: %[[BITS:.+]] = llvm.mlir.constant(-1082130432 : i32) : i32
// CHECK: call @fill_l1_zN_float
// CHECK-SAME: %[[BITS]])
// CHECK-NOT: tla.fill
// CHECK-LABEL: func.func @official_nZ_int16_t()
// CHECK: %[[BITS:.+]] = llvm.mlir.constant(65535 : i32) : i32
// CHECK: call @fill_l1_nZ_int16_t
// CHECK-SAME: %[[BITS]])
// CHECK-NOT: tla.fill
// CHECK-LABEL: func.func @official_nZ_uint16_t()
// CHECK: %[[BITS:.+]] = llvm.mlir.constant(65535 : i32) : i32
// CHECK: call @fill_l1_nZ_uint16_t
// CHECK-SAME: %[[BITS]])
// CHECK-NOT: tla.fill
// CHECK-LABEL: func.func @official_nZ_half()
// CHECK: %[[BITS:.+]] = llvm.mlir.constant(15872 : i32) : i32
// CHECK: call @fill_l1_nZ_half
// CHECK-SAME: %[[BITS]])
// CHECK-NOT: tla.fill
// CHECK-LABEL: func.func @official_nZ_bf16()
// CHECK: %[[BITS:.+]] = llvm.mlir.constant(16320 : i32) : i32
// CHECK: call @fill_l1_nZ_bf16
// CHECK-SAME: %[[BITS]])
// CHECK-NOT: tla.fill
// CHECK-LABEL: func.func @official_nZ_int32_t()
// CHECK: %[[BITS:.+]] = llvm.mlir.constant(-1 : i32) : i32
// CHECK: call @fill_l1_nZ_int32_t
// CHECK-SAME: %[[BITS]])
// CHECK-NOT: tla.fill
// CHECK-LABEL: func.func @official_nZ_uint32_t()
// CHECK: %[[BITS:.+]] = llvm.mlir.constant(-2147483647 : i32) : i32
// CHECK: call @fill_l1_nZ_uint32_t
// CHECK-SAME: %[[BITS]])
// CHECK-NOT: tla.fill
// CHECK-LABEL: func.func @official_nZ_float()
// CHECK: %[[BITS:.+]] = llvm.mlir.constant(-1082130432 : i32) : i32
// CHECK: call @fill_l1_nZ_float
// CHECK-SAME: %[[BITS]])
// CHECK-NOT: tla.fill
