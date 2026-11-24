// RUN: %tla_compile %s -o - | %filecheck %s

// The same nested view read from a SIMT region: the launch GEP must be 96
// elements (6 rows * 16) into the i32 storage, i.e. 384 bytes -- not the 160
// elements (640 bytes) produced when the parent offset is added twice.

!gm = !tla.tensor<!tla.layout<!tla.shape<16,16>, !tla.stride<16,1>, !tla.shape<16,16>, RowMajor>, !tla.coord<0,0>, !tla.ptr<i32, gm, 4>>
!parent = !tla.tensor<!tla.layout<!tla.shape<4,8>, !tla.stride<16,1>, !tla.shape<4,8>, RowMajor>, !tla.coord<4,0>, !tla.ptr<i32, gm, 4>>
!child = !tla.tensor<!tla.layout<!tla.shape<2,4>, !tla.stride<16,1>, !tla.shape<2,4>, RowMajor>, !tla.coord<6,0>, !tla.ptr<i32, gm, 4>>

module {
  func.func @simt_nested_view(%gm: memref<256xi32, #hivm.address_space<gm>>) {
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
    "tla.vector"() ({
      "tla.vec.func"() ({
        %p = tla.tensor_ptr_with_coord %child : !child -> !tla.ptr<i32, gm, 4>
        %a = tla.ptrtoint %p : !tla.ptr<i32, gm, 4> -> i64
        %i = arith.index_cast %a : i64 to index
        %v = tla.simt_load %root[%i] : !gm -> i32
        tla.simt_store %root[%c0], %v : !gm, i32
      }) {mode = "simt", thread_block_dim = array<i64: 1, 1, 1>} : () -> ()
    }) : () -> ()
    return
  }
}

// CHECK: llvm.getelementptr {{%.*}}[96] : (!llvm.ptr<1>) -> !llvm.ptr<1>, i32
// CHECK-NOT: tla.tensor_ptr_with_coord
