// RUN: %tla_compile %s -o - | %filecheck %s

// A tensor read only through t.ptr_with_coord must still be captured. Its
// address comes from the outlined buffer parameter, whose launch pointer
// already carries the view start, so the region needs that parameter even
// though no simt_load / simt_store / print touches the tensor. The capture walk
// therefore has to look at t.ptr_with_coord sources, not just element accesses.
//
// %tail is 4 elements into the same 16-element storage as %full, which keeps
// the two buffer parameters distinguishable (8 vs 16 elements, GEP vs no GEP).

!gm_full = !tla.tensor<!tla.layout<!tla.shape<16>, !tla.stride<1>, !tla.shape<16>, RowMajor>, !tla.coord<0>, !tla.ptr<f32, gm, 4>>
!gm_tail = !tla.tensor<!tla.layout<!tla.shape<8>, !tla.stride<1>, !tla.shape<16>, RowMajor>, !tla.coord<4>, !tla.ptr<f32, gm, 4>>

module {
  func.func @simt_ptr_only(%gm_memref: memref<16xf32, #hivm.address_space<gm>>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %c8 = arith.constant 8 : index
    %c16 = arith.constant 16 : index
    %full = tla.tensor_desc %gm_memref shape [%c1, %c16, %c1, %c1] stride [%c16, %c1, %c1, %c1] origin_shape [%c1, %c16] coord [%c0, %c0] : memref<16xf32, #hivm.address_space<gm>> -> !gm_full
    %tail = tla.tensor_desc %gm_memref shape [%c1, %c8, %c1, %c1] stride [%c8, %c1, %c1, %c1] origin_shape [%c1, %c16] coord [%c0, %c4] : memref<16xf32, #hivm.address_space<gm>> -> !gm_tail
    "tla.vector"() ({
      "tla.vec.func"() ({
        %p = tla.tensor_ptr_with_coord %tail : !tla.tensor<!tla.layout<!tla.shape<8>, !tla.stride<1>, !tla.shape<16>, RowMajor>, !tla.coord<4>, !tla.ptr<f32, gm, 4>> -> !tla.ptr<f32, gm, 4>
        %a = tla.ptrtoint %p : !tla.ptr<f32, gm, 4> -> i64
        %i = arith.index_cast %a : i64 to index
        %v = tla.simt_load %full[%i] : !tla.tensor<!tla.layout<!tla.shape<16>, !tla.stride<1>, !tla.shape<16>, RowMajor>, !tla.coord<0>, !tla.ptr<f32, gm, 4>> -> f32
        tla.simt_store %full[%c0], %v : !tla.tensor<!tla.layout<!tla.shape<16>, !tla.stride<1>, !tla.shape<16>, RowMajor>, !tla.coord<0>, !tla.ptr<f32, gm, 4>>, f32
      }) {mode = "simt", thread_block_dim = array<i64: 1, 1, 1>} : () -> ()
    }) : () -> ()
    return
  }
}

// The op must be fully lowered away.
// CHECK-NOT: tla.tensor_ptr_with_coord

// %tail has no element access inside the region, yet it is a buffer parameter:
// the launch folds its view start (4 elements) into that parameter's pointer.
// CHECK: llvm.getelementptr
// CHECK-LABEL: func.func @simt_ptr_only_vf_simt
// CHECK-SAME:    memref<8xf32, 1>
// CHECK-SAME:    memref<16xf32, 1>

// The view-start address is read off the %tail buffer parameter.
// CHECK: memref.extract_aligned_pointer_as_index %arg0
