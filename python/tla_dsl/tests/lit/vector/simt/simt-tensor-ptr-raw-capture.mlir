// RUN: %tla_compile %s -o - | %filecheck %s

// Raw t.ptr and t.ptr_with_coord on the same non-zero-offset view. t.ptr is the
// raw storage base; t.ptr_with_coord is the view start (4 elements in). The two
// must reach the outlined body as different parameters: the buffer parameter
// carries the view GEP into the launch pointer, the raw pointer parameter does
// not. Reading both addresses back from the wrong parameter would still produce
// well-formed IR, so the checks name the parameters explicitly.

!gm_tail = !tla.tensor<!tla.layout<!tla.shape<8>, !tla.stride<1>, !tla.shape<16>, RowMajor>, !tla.coord<4>, !tla.ptr<f32, gm, 4>>

module {
  func.func @simt_raw_ptr(%gm_memref: memref<16xf32, #hivm.address_space<gm>>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %c8 = arith.constant 8 : index
    %c16 = arith.constant 16 : index
    %tail = tla.tensor_desc %gm_memref shape [%c1, %c8, %c1, %c1] stride [%c8, %c1, %c1, %c1] origin_shape [%c1, %c16] coord [%c0, %c4] : memref<16xf32, #hivm.address_space<gm>> -> !gm_tail
    "tla.vector"() ({
      "tla.vec.func"() ({
        %v = tla.simt_load %tail[%c0] : <!tla.layout<!tla.shape<8>, !tla.stride<1>, !tla.shape<16>, RowMajor>, !tla.coord<4>, !tla.ptr<f32, gm, 4>> -> f32
        %raw = tla.tensor_ptr %tail : !tla.tensor<!tla.layout<!tla.shape<8>, !tla.stride<1>, !tla.shape<16>, RowMajor>, !tla.coord<4>, !tla.ptr<f32, gm, 4>> -> !tla.ptr<f32, gm, 4>
        %ra = tla.ptrtoint %raw : !tla.ptr<f32, gm, 4> -> i64
        %ri = arith.index_cast %ra : i64 to index
        %w = tla.simt_load %tail[%ri] : <!tla.layout<!tla.shape<8>, !tla.stride<1>, !tla.shape<16>, RowMajor>, !tla.coord<4>, !tla.ptr<f32, gm, 4>> -> f32
        %view = tla.tensor_ptr_with_coord %tail : !tla.tensor<!tla.layout<!tla.shape<8>, !tla.stride<1>, !tla.shape<16>, RowMajor>, !tla.coord<4>, !tla.ptr<f32, gm, 4>> -> !tla.ptr<f32, gm, 4>
        %va = tla.ptrtoint %view : !tla.ptr<f32, gm, 4> -> i64
        %vi = arith.index_cast %va : i64 to index
        %x = tla.simt_load %tail[%vi] : <!tla.layout<!tla.shape<8>, !tla.stride<1>, !tla.shape<16>, RowMajor>, !tla.coord<4>, !tla.ptr<f32, gm, 4>> -> f32
        %s0 = tla.simt_add %v, %w : f32
        %s = tla.simt_add %s0, %x : f32
        tla.simt_store %tail[%c0], %s : <!tla.layout<!tla.shape<8>, !tla.stride<1>, !tla.shape<16>, RowMajor>, !tla.coord<4>, !tla.ptr<f32, gm, 4>>, f32
      }) {mode = "simt", thread_block_dim = array<i64: 1, 1, 1>} : () -> ()
    }) : () -> ()
    return
  }
}

// Both pointer ops must be fully lowered away.
// CHECK-NOT: tla.tensor_ptr

// The view start (4 f32 elements) is folded into the buffer launch pointer as
// a GEP, while the raw pointer argument is the un-advanced base. Binding the
// launch operands proves the order: view pointer first, raw base second.
// CHECK: [[RAW:%[a-zA-Z0-9_.]+]] = llvm.inttoptr
// CHECK: [[VIEW:%[a-zA-Z0-9_.]+]] = llvm.getelementptr [[RAW]][{{.*}}]
// CHECK: hivm_regbaseintrins.intrins.launch_func @simt_raw_ptr_vf_simt
// CHECK-SAME: args([[VIEW]], [[RAW]]) : !llvm.ptr<1>, !llvm.ptr<1>

// The buffer parameter spans the view only (8 f32); the raw pointer keeps the
// storage extent (16 f32), which keeps the two parameters distinguishable.
// CHECK-LABEL: func.func @simt_raw_ptr_vf_simt
// CHECK-SAME:    memref<8xf32, 1>
// CHECK-SAME:    memref<16xf32, 1>

// t.ptr reads the raw storage base off the raw-pointer parameter...
// CHECK: memref.extract_aligned_pointer_as_index %arg1
// ...and t.ptr_with_coord reads the view start off the buffer parameter.
// CHECK: memref.extract_aligned_pointer_as_index %arg0
