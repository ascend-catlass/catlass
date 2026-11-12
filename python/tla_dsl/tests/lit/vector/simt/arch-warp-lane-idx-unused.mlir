// RUN: %tla_compile %s --mlir-print-ir-after=tla-vector-region-simt -o %t 2>&1 | %filecheck %s --implicit-check-not=thread_id_ --implicit-check-not=tla.arch.lane_idx

!out = !tla.tensor<!tla.layout<!tla.shape<1>, !tla.stride<1>, !tla.shape<1>, RowMajor>, !tla.coord<0>, !tla.ptr<i32, gm, 4>>
module {
  func.func @lane_query(%mem: memref<1xi32, #hivm.address_space<gm>>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %out = tla.tensor_desc %mem shape [%c1, %c1, %c1, %c1] stride [%c1, %c1, %c1, %c1] origin_shape [%c1, %c1] coord [%c0, %c0] : memref<1xi32, #hivm.address_space<gm>> -> !out
    "tla.vector"() ({
      "tla.vec.func"() ({
        %lane = tla.arch.lane_idx -> i32
        %value = arith.constant 7 : i32
        tla.simt_store %out[%c0], %value : <!tla.layout<!tla.shape<1>, !tla.stride<1>, !tla.shape<1>, RowMajor>, !tla.coord<0>, !tla.ptr<i32, gm, 4>>, i32
      }) {mode = "simt", thread_block_dim = array<i64: 48, 1, 1>} : () -> ()
    }) : () -> ()
    return
  }
}

// CHECK: hivm_regbaseintrins.intrins.launch_func @lane_query_vf_simt
// CHECK-LABEL: func.func @lane_query_vf_simt(
// CHECK: %[[VALUE:.*]] = arith.constant 7 : i32
// CHECK: memref.store %[[VALUE]]
