// RUN: %tla_compile %s --mlir-print-ir-after=tla-vector-region-simt -o %t 2> %t.ir
// RUN: %filecheck %s --check-prefix=ALIGNED --input-file=%t.ir
// RUN: %filecheck %s --check-prefix=THREED --input-file=%t.ir
// RUN: %filecheck %s --check-prefix=LAUNCH --input-file=%t.ir --implicit-check-not=tla.arch.lane_idx

!out = !tla.tensor<!tla.layout<!tla.shape<1>, !tla.stride<1>, !tla.shape<1>, RowMajor>, !tla.coord<0>, !tla.ptr<i32, gm, 4>>
module {
  func.func @two_regions(%mem: memref<1xi32, #hivm.address_space<gm>>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %out = tla.tensor_desc %mem shape [%c1, %c1, %c1, %c1] stride [%c1, %c1, %c1, %c1] origin_shape [%c1, %c1] coord [%c0, %c0] : memref<1xi32, #hivm.address_space<gm>> -> !out
    "tla.vector"() ({
      "tla.vec.func"() ({
        %lane = tla.arch.lane_idx -> i32
        tla.simt_store %out[%c0], %lane : <!tla.layout<!tla.shape<1>, !tla.stride<1>, !tla.shape<1>, RowMajor>, !tla.coord<0>, !tla.ptr<i32, gm, 4>>, i32
      }) {mode = "simt", thread_block_dim = array<i64: 32, 1, 1>} : () -> ()
      "tla.vec.func"() ({
        %other = tla.arch.lane_idx -> i32
        tla.simt_store %out[%c0], %other : <!tla.layout<!tla.shape<1>, !tla.stride<1>, !tla.shape<1>, RowMajor>, !tla.coord<0>, !tla.ptr<i32, gm, 4>>, i32
      }) {mode = "simt", thread_block_dim = array<i64: 4, 4, 3>} : () -> ()
    }) : () -> ()
    return
  }
}

// LAUNCH-COUNT-2: hivm_regbaseintrins.intrins.launch_func @two_regions_vf_simt
// ALIGNED: func.func @two_regions_vf_simt{{[0-9]*}}({{.*}}simt_entry<32>
// ALIGNED: %[[X:.*]] = hivm_regbaseintrins.thread_id_x
// ALIGNED: %[[M:.*]] = arith.constant 31 : i32
// ALIGNED-NOT: thread_id_y
// ALIGNED-NOT: thread_id_z
// ALIGNED: %[[LANE:.*]] = arith.andi %[[X]], %[[M]] : i32
// ALIGNED: memref.store %[[LANE]]
// ALIGNED: return
// THREED: func.func @two_regions_vf_simt{{[0-9]*}}({{.*}}simt_entry<48>
// THREED: %[[X:.*]] = hivm_regbaseintrins.thread_id_x
// THREED: %[[M:.*]] = arith.constant 31 : i32
// THREED: %[[Y:.*]] = hivm_regbaseintrins.thread_id_y
// THREED: %[[BX:.*]] = arith.constant 4 : i32
// THREED: %[[YS:.*]] = arith.muli %[[Y]], %[[BX]] : i32
// THREED: %[[XY:.*]] = arith.addi %[[X]], %[[YS]] : i32
// THREED: %[[Z:.*]] = hivm_regbaseintrins.thread_id_z
// THREED: %[[BXY:.*]] = arith.constant 16 : i32
// THREED: %[[ZS:.*]] = arith.muli %[[Z]], %[[BXY]] : i32
// THREED: %[[XYZ:.*]] = arith.addi %[[XY]], %[[ZS]] : i32
// THREED: %[[LANE:.*]] = arith.andi %[[XYZ]], %[[M]] : i32
// THREED: memref.store %[[LANE]]
// THREED: return
