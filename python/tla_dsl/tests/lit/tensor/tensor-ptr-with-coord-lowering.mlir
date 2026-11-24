// RUN: %tla_compile %s -o - | %filecheck %s

// tla.tensor_ptr_with_coord outside a SIMT region resolves at tla-lower-ptr:
// the backing pointer advanced by coord[0]*stride[0] + coord[1]*stride[1]
// elements -- the same formula the SIMT launch GEP uses, shared through
// descriptorViewElementOffset. Element 4 of an f32 view is 16 bytes.
// tla.tensor_ptr would be the raw base with no offset arithmetic at all.

!gm_tail = !tla.tensor<!tla.layout<!tla.shape<8>, !tla.stride<1>, !tla.shape<16>, RowMajor>, !tla.coord<4>, !tla.ptr<f32, gm, 4>>

module {
  func.func @scalar_ptr_with_coord(%gm_memref: memref<16xf32, #hivm.address_space<gm>>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %c8 = arith.constant 8 : index
    %c16 = arith.constant 16 : index
    %tail = tla.tensor_desc %gm_memref shape [%c1, %c8, %c1, %c1] stride [%c16, %c1, %c1, %c1] origin_shape [%c1, %c16] coord [%c0, %c4] : memref<16xf32, #hivm.address_space<gm>> -> !gm_tail
    %p = tla.tensor_ptr_with_coord %tail : !tla.tensor<!tla.layout<!tla.shape<8>, !tla.stride<1>, !tla.shape<16>, RowMajor>, !tla.coord<4>, !tla.ptr<f32, gm, 4>> -> !tla.ptr<f32, gm, 4>
    %a = tla.ptrtoint %p : !tla.ptr<f32, gm, 4> -> i64
    %i = arith.index_cast %a : i64 to index
    %v = tla.scalar_load %tail[%i] : !tla.tensor<!tla.layout<!tla.shape<8>, !tla.stride<1>, !tla.shape<16>, RowMajor>, !tla.coord<4>, !tla.ptr<f32, gm, 4>> -> f32
    tla.scalar_store %tail[%c1], %v : !tla.tensor<!tla.layout<!tla.shape<8>, !tla.stride<1>, !tla.shape<16>, RowMajor>, !tla.coord<4>, !tla.ptr<f32, gm, 4>>, f32
    return
  }
}

// The op must be fully lowered away.
// CHECK-NOT: tla.tensor_ptr_with_coord

// The view start (4 f32 elements) is scaled into a 16-byte address offset.
// CHECK: llvm.mlir.constant(16 : i64)
