// RUN: %tla_compile %s -o - | %filecheck %s

// A view sliced twice: parent starts at row 4, the child moves 2 rows further.
// The child's start is row 6, so its pointer is 6 * 16 * 4 = 384 bytes past the
// base. The derivation must add the parent offset once (from the child
// tile_view's relative coord operand), not twice (relative operand plus the
// already-accumulated coord in the child's tensor type).

!gm = !tla.tensor<!tla.layout<!tla.shape<16,16>, !tla.stride<16,1>, !tla.shape<16,16>, RowMajor>, !tla.coord<0,0>, !tla.ptr<i32, gm, 4>>
!parent = !tla.tensor<!tla.layout<!tla.shape<4,8>, !tla.stride<16,1>, !tla.shape<4,8>, RowMajor>, !tla.coord<4,0>, !tla.ptr<i32, gm, 4>>
!child = !tla.tensor<!tla.layout<!tla.shape<2,4>, !tla.stride<16,1>, !tla.shape<2,4>, RowMajor>, !tla.coord<6,0>, !tla.ptr<i32, gm, 4>>

module {
  func.func @nested_view(%gm: memref<256xi32, #hivm.address_space<gm>>, %sink: memref<1xi64>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c4 = arith.constant 4 : index
    %c8 = arith.constant 8 : index
    %c16 = arith.constant 16 : index
    %root = tla.tensor_desc %gm shape [%c16, %c16, %c1, %c1] stride [%c16, %c1, %c1, %c1] origin_shape [%c16, %c16] coord [%c0, %c0] : memref<256xi32, #hivm.address_space<gm>> -> !gm
    %pshape = tla.make_shape -> !tla.shape<4,8>
    %pcoord = tla.make_coord -> !tla.coord<4,0>
    %parent = tla.tile_view %root, %pshape, %pcoord : !gm, !tla.shape<4,8>, !tla.coord<4,0> -> !parent
    %cshape = tla.make_shape -> !tla.shape<2,4>
    %ccoord = tla.make_coord -> !tla.coord<2,0>
    %child = tla.tile_view %parent, %cshape, %ccoord : !parent, !tla.shape<2,4>, !tla.coord<2,0> -> !child
    %p = tla.tensor_ptr_with_coord %child : !child -> !tla.ptr<i32, gm, 4>
    %a = tla.ptrtoint %p : !tla.ptr<i32, gm, 4> -> i64
    memref.store %a, %sink[%c0] : memref<1xi64>
    return
  }
}

// The view start is 6 rows in: 6 * 16 elements * 4 bytes = 384.
// CHECK: llvm.mlir.constant(384 : i64)
// CHECK: llvm.add
// CHECK-NOT: tla.tensor_ptr_with_coord
