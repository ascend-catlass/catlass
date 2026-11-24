// RUN: %tla_compile %s -o - | %filecheck %s

// A raw t.ptr read from a SIMT region whose backing memref is strided. The
// pointer capture only needs the storage address
// (extract_aligned_pointer_as_index), never an index, so a strided layout is
// fine here: it crosses as its own bare-pointer launch argument with no view
// GEP, and the parameter's identity layout is only ever asked for its address.
// A region that *indexed* such a memref directly would be rejected instead --
// see simt-vec-func-nonscalar-capture.mlir.

!gm_full = !tla.tensor<!tla.layout<!tla.shape<8>, !tla.stride<1>, !tla.shape<8>, RowMajor>, !tla.coord<0>, !tla.ptr<f32, gm, 4>>

module {
  func.func @simt_raw_ptr_strided(%gm: memref<8xf32, strided<[2]>, #hivm.address_space<gm>>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c8 = arith.constant 8 : index
    %full = tla.tensor_desc %gm shape [%c1, %c8, %c1, %c1] stride [%c8, %c1, %c1, %c1] origin_shape [%c1, %c8] coord [%c0, %c0] : memref<8xf32, strided<[2]>, #hivm.address_space<gm>> -> !gm_full
    "tla.vector"() ({
      "tla.vec.func"() ({
        %v = tla.simt_load %full[%c0] : <!tla.layout<!tla.shape<8>, !tla.stride<1>, !tla.shape<8>, RowMajor>, !tla.coord<0>, !tla.ptr<f32, gm, 4>> -> f32
        %p = tla.tensor_ptr %full : !tla.tensor<!tla.layout<!tla.shape<8>, !tla.stride<1>, !tla.shape<8>, RowMajor>, !tla.coord<0>, !tla.ptr<f32, gm, 4>> -> !tla.ptr<f32, gm, 4>
        %a = tla.ptrtoint %p : !tla.ptr<f32, gm, 4> -> i64
        %i = arith.index_cast %a : i64 to index
        %w = tla.simt_load %full[%i] : <!tla.layout<!tla.shape<8>, !tla.stride<1>, !tla.shape<8>, RowMajor>, !tla.coord<0>, !tla.ptr<f32, gm, 4>> -> f32
        %s = tla.simt_add %v, %w : f32
        tla.simt_store %full[%c1], %s : <!tla.layout<!tla.shape<8>, !tla.stride<1>, !tla.shape<8>, RowMajor>, !tla.coord<0>, !tla.ptr<f32, gm, 4>>, f32
      }) {mode = "simt", thread_block_dim = array<i64: 1, 1, 1>} : () -> ()
    }) : () -> ()
    return
  }
}

// The op must be fully lowered away.
// CHECK-NOT: tla.tensor_ptr

// The buffer pointer and the raw pointer are separate launch arguments, both
// !llvm.ptr<1>, even though the backing memref is strided.
// CHECK: hivm_regbaseintrins.intrins.launch_func @simt_raw_ptr_strided_vf_simt
// CHECK-SAME: args({{.*}}) : !llvm.ptr<1>, !llvm.ptr<1>

// The outlined body reads the raw base address off the pointer parameter, which
// is the second one (%arg1); the buffer parameter (%arg0) is only indexed.
// CHECK-LABEL: func.func @simt_raw_ptr_strided_vf_simt
// CHECK-SAME:    memref<8xf32, 1>
// CHECK-SAME:    memref<8xf32, 1>
// CHECK: memref.extract_aligned_pointer_as_index %arg1
