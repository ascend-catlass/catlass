// RUN: %tla_compile %s -o - | %filecheck %s

// Scalar element access on a twice-sliced view. The descriptor's absolute coord
// must be row 6 (parent row 4 + child row 2), so the scalar view starts at
// 6 * 16 = 96 elements. Reading the child tile_view's accumulated result coord
// and adding the parent coord again produced 160.

!gm = !tla.tensor<!tla.layout<!tla.shape<16,16>, !tla.stride<16,1>, !tla.shape<16,16>, RowMajor>, !tla.coord<0,0>, !tla.ptr<i32, gm, 4>>
!parent = !tla.tensor<!tla.layout<!tla.shape<4,8>, !tla.stride<16,1>, !tla.shape<4,8>, RowMajor>, !tla.coord<4,0>, !tla.ptr<i32, gm, 4>>
!child = !tla.tensor<!tla.layout<!tla.shape<2,4>, !tla.stride<16,1>, !tla.shape<2,4>, RowMajor>, !tla.coord<6,0>, !tla.ptr<i32, gm, 4>>

module {
  func.func @nested_scalar(%gm: memref<256xi32, #hivm.address_space<gm>>, %sink: memref<1xi32>) {
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
    %v = tla.scalar_load %child[%c0, %c0] : !child -> i32
    memref.store %v, %sink[%c0] : memref<1xi32>
    return
  }
}

// CHECK: llvm.mlir.constant(96 : index)
// CHECK: memref.reinterpret_cast
