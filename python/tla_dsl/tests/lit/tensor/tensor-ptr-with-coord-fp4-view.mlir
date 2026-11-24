// RUN: %tla_compile %s -o - | %filecheck %s

// A Float4E2M1 view starting at logical element 32. Two fp4 elements share one
// byte, so the byte address is 32 / 2 = 16 past the base. Element offsets on
// packed sub-byte pointers must be scaled by the logical bit width, not by a
// byte width the type does not have. The pointer feeds
// tla.arch.dcache_prefetch, the consumer that previously failed to compile
// because the internal ptr_add had no byte width to scale by.

!gm = !tla.tensor<!tla.layout<!tla.shape<16,16>, !tla.stride<16,1>, !tla.shape<16,16>, RowMajor>, !tla.coord<0,0>, !tla.ptr<!tla.f4e2m1, gm, 1>>
!view = !tla.tensor<!tla.layout<!tla.shape<2,8>, !tla.stride<16,1>, !tla.shape<2,8>, RowMajor>, !tla.coord<2,0>, !tla.ptr<!tla.f4e2m1, gm, 1>>

module {
  func.func @fp4_view(%gm: memref<128xi8, #hivm.address_space<gm>>, %sink: memref<1xi64>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c8 = arith.constant 8 : index
    %c16 = arith.constant 16 : index
    %root = tla.tensor_desc %gm shape [%c16, %c16, %c1, %c1] stride [%c16, %c1, %c1, %c1] origin_shape [%c16, %c16] coord [%c0, %c0] : memref<128xi8, #hivm.address_space<gm>> -> !gm
    %vs = tla.make_shape -> !tla.shape<2,8>
    %vc = tla.make_coord -> !tla.coord<2,0>
    %view = tla.tile_view %root, %vs, %vc : !gm, !tla.shape<2,8>, !tla.coord<2,0> -> !view
    %p = tla.tensor_ptr_with_coord %view : !view -> !tla.ptr<!tla.f4e2m1, gm, 1>
    %c0_64 = arith.constant 0 : i64
    "tla.vector"() ({
      "tla.arch.dcache_prefetch"(%p, %c0_64) : (!tla.ptr<!tla.f4e2m1, gm, 1>, i64) -> ()
    }) : () -> ()
    %a = tla.ptrtoint %p : !tla.ptr<!tla.f4e2m1, gm, 1> -> i64
    memref.store %a, %sink[%c0] : memref<1xi64>
    return
  }
}

// 32 logical fp4 elements are 16 bytes.
// CHECK: llvm.mlir.constant(16 : i64)
// CHECK: llvm.add
// The prefetch consumer sees the same scaled address.
// CHECK: call @_mlir_ciface_tla_dcache_prefetch
// CHECK-NOT: tla.ptr_add
