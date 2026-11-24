// RUN: %tla_compile %s -o - | %filecheck %s

// Selecting a view by runtime control flow cannot be resolved after the fact:
// t.ptr on the scf.if result cannot trace its buffer. The supported spelling is
// to take ptr_with_coord inside each branch and select the resulting pointers,
// which is exactly what this test does.

!gm = !tla.tensor<!tla.layout<!tla.shape<16,16>, !tla.stride<16,1>, !tla.shape<16,16>, RowMajor>, !tla.coord<0,0>, !tla.ptr<i32, gm, 4>>
!va = !tla.tensor<!tla.layout<!tla.shape<4,8>, !tla.stride<16,1>, !tla.shape<4,8>, RowMajor>, !tla.coord<2,0>, !tla.ptr<i32, gm, 4>>
!vb = !tla.tensor<!tla.layout<!tla.shape<4,8>, !tla.stride<16,1>, !tla.shape<4,8>, RowMajor>, !tla.coord<1,0>, !tla.ptr<i32, gm, 4>>

module {
  func.func @scf_branch_pointers(%gm: memref<256xi32, #hivm.address_space<gm>>, %sink: memref<1xi64>, %raw_sink: memref<1xi64>, %cond: i1) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c4 = arith.constant 4 : index
    %c8 = arith.constant 8 : index
    %c16 = arith.constant 16 : index
    %root = tla.tensor_desc %gm shape [%c16, %c16, %c1, %c1] stride [%c16, %c1, %c1, %c1] origin_shape [%c16, %c16] coord [%c0, %c0] : memref<256xi32, #hivm.address_space<gm>> -> !gm
    %as = tla.make_shape -> !tla.shape<4,8>
    %ac = tla.make_coord -> !tla.coord<2,0>
    %viewa = tla.tile_view %root, %as, %ac : !gm, !tla.shape<4,8>, !tla.coord<2,0> -> !va
    %bc = tla.make_coord -> !tla.coord<1,0>
    %viewb = tla.tile_view %root, %as, %bc : !gm, !tla.shape<4,8>, !tla.coord<1,0> -> !vb
    %pa = tla.tensor_ptr_with_coord %viewa : !va -> !tla.ptr<i32, gm, 4>
    %pb = tla.tensor_ptr_with_coord %viewb : !vb -> !tla.ptr<i32, gm, 4>
    %p = scf.if %cond -> (!tla.ptr<i32, gm, 4>) {
      scf.yield %pa : !tla.ptr<i32, gm, 4>
    } else {
      scf.yield %pb : !tla.ptr<i32, gm, 4>
    }
    %rawa = tla.tensor_ptr %viewa : !va -> !tla.ptr<i32, gm, 4>
    %rawb = tla.tensor_ptr %viewb : !vb -> !tla.ptr<i32, gm, 4>
    %raw = scf.if %cond -> (!tla.ptr<i32, gm, 4>) {
      scf.yield %rawa : !tla.ptr<i32, gm, 4>
    } else {
      scf.yield %rawb : !tla.ptr<i32, gm, 4>
    }
    %a = tla.ptrtoint %p : !tla.ptr<i32, gm, 4> -> i64
    %raw_a = tla.ptrtoint %raw : !tla.ptr<i32, gm, 4> -> i64
    memref.store %a, %sink[%c0] : memref<1xi64>
    memref.store %raw_a, %raw_sink[%c0] : memref<1xi64>
    return
  }
}

// Row 2 is 128 bytes in, row 1 is 64 bytes in; both pointers are selected at
// run time.
// CHECK-DAG: llvm.mlir.constant(128 : i64)
// CHECK-DAG: llvm.mlir.constant(64 : i64)
// Both selections branch at run time: the view-start pointers and the raw base
// pointers each come from their own scf.if.
// CHECK-COUNT-2: cf.cond_br
// CHECK-NOT: tla.tensor_ptr
