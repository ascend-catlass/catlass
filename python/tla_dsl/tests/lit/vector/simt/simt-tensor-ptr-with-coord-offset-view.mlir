// RUN: %tla_compile %s -o - | %filecheck %s

// tla.tensor_ptr_with_coord read from a SIMT region. The view start (element 4)
// is folded into the buffer's launch pointer already, so the outlined body
// reads the address off the buffer parameter -- the op costs no extra launch
// argument. t.ptr would pass the un-advanced base instead; see
// simt-tensor-ptr-raw-capture.mlir.

!ub_full = !tla.tensor<!tla.layout<!tla.shape<16>, !tla.stride<1>, !tla.shape<16>, RowMajor>, !tla.coord<0>, !tla.ptr<f32, ub, 4>>
!ub_tail = !tla.tensor<!tla.layout<!tla.shape<8>, !tla.stride<1>, !tla.shape<16>, RowMajor>, !tla.coord<4>, !tla.ptr<f32, ub, 4>>

module {
  func.func @simt_ptr_with_coord(%ub_memref: memref<16xf32, #hivm.address_space<ub>>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %c8 = arith.constant 8 : index
    %c16 = arith.constant 16 : index
    %full = tla.tensor_desc %ub_memref shape [%c1, %c16, %c1, %c1] stride [%c16, %c1, %c1, %c1] origin_shape [%c1, %c16] coord [%c0, %c0] : memref<16xf32, #hivm.address_space<ub>> -> !ub_full
    %tail = tla.tensor_desc %ub_memref shape [%c1, %c8, %c1, %c1] stride [%c16, %c1, %c1, %c1] origin_shape [%c1, %c16] coord [%c0, %c4] : memref<16xf32, #hivm.address_space<ub>> -> !ub_tail
    "tla.vector"() ({
      "tla.vec.func"() ({
        %v = tla.simt_load %full[%c0] : <!tla.layout<!tla.shape<16>, !tla.stride<1>, !tla.shape<16>, RowMajor>, !tla.coord<0>, !tla.ptr<f32, ub, 4>> -> f32
        tla.simt_store %tail[%c0], %v : <!tla.layout<!tla.shape<8>, !tla.stride<1>, !tla.shape<16>, RowMajor>, !tla.coord<4>, !tla.ptr<f32, ub, 4>>, f32
        %p = tla.tensor_ptr_with_coord %tail : !tla.tensor<!tla.layout<!tla.shape<8>, !tla.stride<1>, !tla.shape<16>, RowMajor>, !tla.coord<4>, !tla.ptr<f32, ub, 4>> -> !tla.ptr<f32, ub, 4>
        %a = tla.ptrtoint %p : !tla.ptr<f32, ub, 4> -> i64
        %i = arith.index_cast %a : i64 to index
        %w = tla.simt_load %tail[%i] : <!tla.layout<!tla.shape<8>, !tla.stride<1>, !tla.shape<16>, RowMajor>, !tla.coord<4>, !tla.ptr<f32, ub, 4>> -> f32
        %s = tla.simt_add %v, %w : f32
        tla.simt_store %full[%c1], %s : <!tla.layout<!tla.shape<16>, !tla.stride<1>, !tla.shape<16>, RowMajor>, !tla.coord<0>, !tla.ptr<f32, ub, 4>>, f32
      }) {mode = "simt", thread_block_dim = array<i64: 1, 1, 1>} : () -> ()
    }) : () -> ()
    return
  }
}

// The op must be fully lowered away.
// CHECK-NOT: tla.tensor_ptr_with_coord

// Two buffer parameters only: no extra launch argument for the pointer. The
// full-view parameter is the raw base; the tail parameter (coord 4) carries the
// view start as a GEP by 4 elements.
// CHECK: [[BASE:%[a-zA-Z0-9_.]+]] = llvm.inttoptr
// CHECK: [[VIEW:%[a-zA-Z0-9_.]+]] = llvm.getelementptr [[BASE]][{{.*}}]
// CHECK: hivm_regbaseintrins.intrins.launch_func @simt_ptr_with_coord_vf_simt
// CHECK-SAME: args([[BASE]], [[VIEW]]) : !llvm.ptr<6>, !llvm.ptr<6>
// CHECK-LABEL: func.func @simt_ptr_with_coord_vf_simt
// CHECK-SAME:    memref<16xf32, 6>
// CHECK-SAME:    memref<8xf32, 6>

// The view-start address is read off the tail buffer parameter (%arg1), not off
// the full-view one; the address->ptr->int round trip folds straight back to
// that parameter's base.
// CHECK: memref.extract_aligned_pointer_as_index %arg1
