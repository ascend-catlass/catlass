// RUN: %tla_compile %s --mlir-print-ir-after=tla-vector-region-simt -o %t 2>&1 | %filecheck %s --implicit-check-not=thread_id_z

!out = !tla.tensor<!tla.layout<!tla.shape<1>, !tla.stride<1>, !tla.shape<1>, RowMajor>, !tla.coord<0>, !tla.ptr<i32, gm, 4>>
module {
  func.func @lane_query(%mem: memref<1xi32, #hivm.address_space<gm>>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %out = tla.tensor_desc %mem shape [%c1, %c1, %c1, %c1] stride [%c1, %c1, %c1, %c1] origin_shape [%c1, %c1] coord [%c0, %c0] : memref<1xi32, #hivm.address_space<gm>> -> !out
    "tla.vector"() ({
      "tla.vec.func"() ({
        %lane = tla.arch.lane_idx -> i32
        tla.simt_store %out[%c0], %lane : <!tla.layout<!tla.shape<1>, !tla.stride<1>, !tla.shape<1>, RowMajor>, !tla.coord<0>, !tla.ptr<i32, gm, 4>>, i32
      }) {mode = "simt", thread_block_dim = array<i64: 8, 4, 1>} : () -> ()
    }) : () -> ()
    return
  }
}

// CHECK: hivm_regbaseintrins.intrins.launch_func @lane_query_vf_simt
// CHECK-SAME: !llvm.ptr<1>
// CHECK-LABEL: func.func @lane_query_vf_simt(
// CHECK-SAME: memref<1xi32, 1>
// CHECK-SAME: hivm_regbaseintrins.cconv = #hivm_regbaseintrins.simt_entry<32>
// CHECK: %[[TX:.*]] = hivm_regbaseintrins.thread_id_x
// CHECK: %[[MASK:.*]] = arith.constant 31 : i32
// CHECK: %[[TY:.*]] = hivm_regbaseintrins.thread_id_y
// CHECK: %[[BX:.*]] = arith.constant 8 : i32
// CHECK: %[[Y:.*]] = arith.muli %[[TY]], %[[BX]] : i32
// CHECK: %[[XY:.*]] = arith.addi %[[TX]], %[[Y]] : i32
// CHECK: %[[LANE:.*]] = arith.andi %[[XY]], %[[MASK]] : i32
// CHECK: memref.store %[[LANE]]
// CHECK-NOT: tla.arch.lane_idx
