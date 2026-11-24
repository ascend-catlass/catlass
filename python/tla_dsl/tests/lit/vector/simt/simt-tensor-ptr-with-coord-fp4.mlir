// RUN: %tla_compile %s -o - | %filecheck %s

// A Float4E2M1 view starting at logical element 32 read from a SIMT region. The
// launch GEP steps in storage bytes (the buffer parameter is i8), so the
// element offset must first be divided by two: GEP [16], i.e. 16 bytes. The
// buggy path emitted GEP [32] and advanced 32 bytes on device.

!gm = !tla.tensor<!tla.layout<!tla.shape<16,16>, !tla.stride<16,1>, !tla.shape<16,16>, RowMajor>, !tla.coord<0,0>, !tla.ptr<!tla.f4e2m1, gm, 1>>
!view = !tla.tensor<!tla.layout<!tla.shape<2,8>, !tla.stride<16,1>, !tla.shape<2,8>, RowMajor>, !tla.coord<2,0>, !tla.ptr<!tla.f4e2m1, gm, 1>>
!sink = !tla.tensor<!tla.layout<!tla.shape<16>, !tla.stride<1>, !tla.shape<16>, RowMajor>, !tla.coord<0>, !tla.ptr<i32, gm, 4>>

module {
  func.func @simt_fp4_view(%gm: memref<128xi8, #hivm.address_space<gm>>, %out: memref<16xi32, #hivm.address_space<gm>>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c8 = arith.constant 8 : index
    %c16 = arith.constant 16 : index
    %root = tla.tensor_desc %gm shape [%c16, %c16, %c1, %c1] stride [%c16, %c1, %c1, %c1] origin_shape [%c16, %c16] coord [%c0, %c0] : memref<128xi8, #hivm.address_space<gm>> -> !gm
    %vs = tla.make_shape -> !tla.shape<2,8>
    %vc = tla.make_coord -> !tla.coord<2,0>
    %view = tla.tile_view %root, %vs, %vc : !gm, !tla.shape<2,8>, !tla.coord<2,0> -> !view
    %s = tla.tensor_desc %out shape [%c1, %c16, %c1, %c1] stride [%c16, %c1, %c1, %c1] origin_shape [%c1, %c16] coord [%c0, %c0] : memref<16xi32, #hivm.address_space<gm>> -> !sink
    "tla.vector"() ({
      "tla.vec.func"() ({
        %p = tla.tensor_ptr_with_coord %view : !view -> !tla.ptr<!tla.f4e2m1, gm, 1>
        %a = tla.ptrtoint %p : !tla.ptr<!tla.f4e2m1, gm, 1> -> i64
        %i = arith.index_cast %a : i64 to index
        %v = tla.simt_load %s[%i] : !sink -> i32
        tla.simt_store %s[%c0], %v : !sink, i32
      }) {mode = "simt", thread_block_dim = array<i64: 1, 1, 1>} : () -> ()
    }) : () -> ()
    return
  }
}

// The view starts 32 logical elements (16 bytes) into the packed storage.
// CHECK: llvm.getelementptr {{%.*}}[16] : (!llvm.ptr<1>) -> !llvm.ptr<1>, i8
// CHECK-NOT: tla.tensor_ptr_with_coord
