// RUN: %tla_compile %s -o - --mlir-print-ir-after=tla-insert-auto-mutex > %t 2>&1
// RUN: %filecheck %s --check-prefix=CHECK < %t
// RUN: %filecheck %s --check-prefix=F32 < %t
//
// tla.fill takes part in auto-mutex like the copy whose tile it pads: the bc
// helper is an MTE2 instruction writing L1, so the destination tile is locked
// with the <mte2> pipe for the duration of the fill. The fill reads nothing,
// so the destination is its only resource.

module {
  tla.func @auto_mutex_fill() attributes {tla.auto_sync = "v0"} {
    %raw = tla.alloc_ptr{size_bytes = 4096} -> !tla.ptr<i8, l1, 512>
    %ptr = tla.recast_ptr %raw : !tla.ptr<i8, l1, 512> -> !tla.ptr<f8E4M3FN, l1, 512>
    %shape = tla.make_shape -> !tla.shape<(16,2),(32,2)>
    %stride = tla.make_stride -> !tla.stride<(32,512),(1,1024)>
    %origin = tla.make_shape -> !tla.shape<32,64>
    %layout = tla.make_layout %shape, %stride origin %origin {layoutTag = #tla.layout_tag<zN>} :
      !tla.shape<(16,2),(32,2)>, !tla.stride<(32,512),(1,1024)> origin !tla.shape<32,64> ->
      !tla.layout<!tla.shape<(16,2),(32,2)>, !tla.stride<(32,512),(1,1024)>, !tla.shape<32,64>, zN>
    %zero_coord = tla.make_coord -> !tla.coord<0,0>
    %dst = tla.make_tensor %ptr, %layout, %zero_coord :
      !tla.ptr<f8E4M3FN, l1, 512>,
      !tla.layout<!tla.shape<(16,2),(32,2)>, !tla.stride<(32,512),(1,1024)>, !tla.shape<32,64>, zN>,
      !tla.coord<0,0> ->
      !tla.tensor<!tla.layout<!tla.shape<(16,2),(32,2)>, !tla.stride<(32,512),(1,1024)>, !tla.shape<32,64>, zN>, !tla.coord<0,0>, !tla.ptr<f8E4M3FN, l1, 512>>
    %zero = arith.constant 0 : i32
    %fill_coord = tla.make_coord -> !tla.coord<0,32>
    %fill_shape = tla.make_shape -> !tla.shape<32,32>
    "tla.cube"() ({
      tla.fill %dst, %zero, %fill_coord, %fill_shape :
        !tla.tensor<!tla.layout<!tla.shape<(16,2),(32,2)>, !tla.stride<(32,512),(1,1024)>, !tla.shape<32,64>, zN>, !tla.coord<0,0>, !tla.ptr<f8E4M3FN, l1, 512>>,
        i32, !tla.coord<0,32>, !tla.shape<32,32>
    }) : () -> ()
    tla.return
  }

  tla.func @auto_mutex_fill_f32() attributes {tla.auto_sync = "v0"} {
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
    %zero = arith.constant 0 : i32
    %fill_coord = tla.make_coord -> !tla.coord<0,32>
    %fill_shape = tla.make_shape -> !tla.shape<32,32>
    "tla.cube"() ({
      tla.fill %dst, %zero, %fill_coord, %fill_shape :
        !tla.tensor<!tla.layout<!tla.shape<(16,2),(8,8)>, !tla.stride<(8,128),(1,256)>, !tla.shape<32,64>, zN>, !tla.coord<0,0>, !tla.ptr<f32, l1, 512>>,
        i32, !tla.coord<0,32>, !tla.shape<32,32>
    }) : () -> ()
    tla.return
  }
}

// CHECK-LABEL: func.func @auto_mutex_fill
// CHECK-NOT: tla.auto_sync
// CHECK: %[[L1:.*]] = tla.mutex "auto_l1_0_4096" {id = 0 : i64} -> !tla.mutex
// CHECK: tla.mutex_lock %[[L1]][<mte2>] : !tla.mutex
// CHECK-NEXT: tla.fill
// CHECK-NEXT: tla.mutex_unlock %[[L1]][<mte2>] : !tla.mutex

// F32-LABEL: func.func @auto_mutex_fill_f32
// F32-NOT: tla.auto_sync
// F32: %[[L1:.*]] = tla.mutex "auto_l1_4096_4096" {id = 0 : i64} -> !tla.mutex
// MTE2: the fill locks the L1 tile it pads.
// F32: tla.mutex_lock %[[L1]][<mte2>] : !tla.mutex
// F32-NEXT: tla.fill
// F32-NEXT: tla.mutex_unlock %[[L1]][<mte2>] : !tla.mutex
