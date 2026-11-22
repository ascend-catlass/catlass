// RUN: %tla_compile %s --mlir-print-ir-after=tla-vector-region-simt -o %t 2>&1 | %filecheck %s --implicit-check-not=thread_id_y --implicit-check-not=thread_id_z

!out = !tla.tensor<!tla.layout<!tla.shape<3>, !tla.stride<1>, !tla.shape<3>, RowMajor>, !tla.coord<0>, !tla.ptr<i32, gm, 4>>
module {
  func.func @active_query(%mem: memref<3xi32, #hivm.address_space<gm>>, %take: i1, %nested: i1) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index
    %out = tla.tensor_desc %mem shape [%c3, %c1, %c1, %c1] stride [%c1, %c1, %c1, %c1] origin_shape [%c3, %c1] coord [%c0, %c0] : memref<3xi32, #hivm.address_space<gm>> -> !out
    "tla.vector"() ({
      "tla.vec.func"() ({
        %outer = tla.arch.activemask -> i32
        tla.simt_store %out[%c0], %outer : <!tla.layout<!tla.shape<3>, !tla.stride<1>, !tla.shape<3>, RowMajor>, !tla.coord<0>, !tla.ptr<i32, gm, 4>>, i32
        scf.if %take {
          %lane = tla.arch.activemask -> i32
          tla.simt_store %out[%c1], %lane : <!tla.layout<!tla.shape<3>, !tla.stride<1>, !tla.shape<3>, RowMajor>, !tla.coord<0>, !tla.ptr<i32, gm, 4>>, i32
          scf.if %nested {
            %inner = tla.arch.activemask -> i32
            tla.simt_store %out[%c2], %inner : <!tla.layout<!tla.shape<3>, !tla.stride<1>, !tla.shape<3>, RowMajor>, !tla.coord<0>, !tla.ptr<i32, gm, 4>>, i32
          }
        }
        %unused = tla.arch.activemask -> i32
      }) {mode = "simt", thread_block_dim = array<i64: 48, 1, 1>} : () -> ()
    }) : () -> ()
    return
  }
}

// CHECK: hivm_regbaseintrins.intrins.launch_func @active_query_vf_simt
// CHECK-SAME: !llvm.ptr<1>, i1, i1
// CHECK-LABEL: func.func @active_query_vf_simt(
// CHECK-SAME: memref<3xi32, 1>
// CHECK-SAME: i1
// CHECK-SAME: hivm_regbaseintrins.cconv = #hivm_regbaseintrins.simt_entry<48>
// Each query lowers at its original site, including both nested branches.
// CHECK-NOT: scf.if
// CHECK: %[[OUTER:.*]] = {{.*}}hivm.redux.add.u32
// CHECK: memref.store %[[OUTER]]
// CHECK: scf.if
// CHECK: %[[TX:.*]] = hivm_regbaseintrins.thread_id_x
// CHECK: %[[MASK:.*]] = arith.constant 31 : i32
// CHECK: %[[LANE:.*]] = arith.andi %[[TX]], %[[MASK]] : i32
// CHECK: %[[ONE:.*]] = arith.constant 1 : i32
// CHECK: %[[BIT:.*]] = arith.shli %[[ONE]], %[[LANE]] : i32
// CHECK: %[[ACTIVE:.*]] = {{.*}}hivm.redux.add.u32 %[[BIT]]
// CHECK: memref.store %[[ACTIVE]]
// CHECK: scf.if
// CHECK: %[[INNER:.*]] = {{.*}}hivm.redux.add.u32
// CHECK: memref.store %[[INNER]]
// CHECK-NOT: tla.arch.activemask
// CHECK-NOT: hivm.redux.add.u32
