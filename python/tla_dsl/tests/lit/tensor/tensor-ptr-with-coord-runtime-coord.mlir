// RUN: %tla_compile %s -o - | %filecheck %s

// tla.tensor_ptr_with_coord on a plain tile_view whose coordinate is a runtime
// value: the tensor type only says `?`, so the view-start offset has to come
// from the tile_view's coord operand (and the parent stride), not from the
// type. The offset is row * 16 elements, scaled to 4-byte f32 addresses.

!gm_full = !tla.tensor<!tla.layout<!tla.shape<16,16>, !tla.stride<16,1>, !tla.shape<16,16>, RowMajor>, !tla.coord<0,0>, !tla.ptr<f32, gm, 4>>
!gm_row = !tla.tensor<!tla.layout<!tla.shape<8,16>, !tla.stride<16,1>, !tla.shape<8,16>, RowMajor>, !tla.coord<?,0>, !tla.ptr<f32, gm, 4>>

module {
  func.func @ptr_runtime(%gm: memref<256xf32, #hivm.address_space<gm>>, %sink: memref<1xi64>, %row: index) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c8 = arith.constant 8 : index
    %c16 = arith.constant 16 : index
    %full = tla.tensor_desc %gm shape [%c16, %c16, %c1, %c1] stride [%c16, %c1, %c1, %c1] origin_shape [%c16, %c16] coord [%c0, %c0] : memref<256xf32, #hivm.address_space<gm>> -> !gm_full
    %ts = tla.make_shape -> !tla.shape<8,16>
    %tc = tla.make_coord %row -> !tla.coord<?,0>
    %view = tla.tile_view %full, %ts, %tc : !gm_full, !tla.shape<8,16>, !tla.coord<?,0> -> !gm_row
    %p = tla.tensor_ptr_with_coord %view : !gm_row -> !tla.ptr<f32, gm, 4>
    %a = tla.ptrtoint %p : !tla.ptr<f32, gm, 4> -> i64
    memref.store %a, %sink[%c0] : memref<1xi64>
    return
  }
}

// The op must be fully lowered away.
// CHECK-NOT: tla.tensor_ptr_with_coord

// The row coordinate is multiplied by the row stride (16 elements) and then by
// the 4-byte element width, so the address carries the runtime view start.
// CHECK-DAG: llvm.mlir.constant(16 : index)
// CHECK-DAG: llvm.mlir.constant(4 : i64)
// CHECK: llvm.mul
// CHECK: llvm.mul
// CHECK: llvm.add
