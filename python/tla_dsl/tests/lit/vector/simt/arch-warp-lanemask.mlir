// RUN: %tla_compile %s --mlir-print-ir-after=tla-vector-region-simt -o %t 2>&1 | %filecheck %s --implicit-check-not=tla.arch.lanemask --implicit-check-not=arith.addi --implicit-check-not="arith.constant 32 : i32"

!out = !tla.tensor<!tla.layout<!tla.shape<5>, !tla.stride<1>, !tla.shape<5>, RowMajor>, !tla.coord<0>, !tla.ptr<i32, gm, 4>>
module {
  func.func @lane_masks(%mem: memref<5xi32, #hivm.address_space<gm>>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index
    %c4 = arith.constant 4 : index
    %c5 = arith.constant 5 : index
    %out = tla.tensor_desc %mem shape [%c5, %c1, %c1, %c1] stride [%c1, %c1, %c1, %c1] origin_shape [%c5, %c1] coord [%c0, %c0] : memref<5xi32, #hivm.address_space<gm>> -> !out
    "tla.vector"() ({
      "tla.vec.func"() ({
        %eq = tla.arch.lanemask "eq" : i32
        tla.simt_store %out[%c0], %eq : <!tla.layout<!tla.shape<5>, !tla.stride<1>, !tla.shape<5>, RowMajor>, !tla.coord<0>, !tla.ptr<i32, gm, 4>>, i32
        %lt = tla.arch.lanemask "lt" : i32
        tla.simt_store %out[%c1], %lt : <!tla.layout<!tla.shape<5>, !tla.stride<1>, !tla.shape<5>, RowMajor>, !tla.coord<0>, !tla.ptr<i32, gm, 4>>, i32
        %le = tla.arch.lanemask "le" : i32
        tla.simt_store %out[%c2], %le : <!tla.layout<!tla.shape<5>, !tla.stride<1>, !tla.shape<5>, RowMajor>, !tla.coord<0>, !tla.ptr<i32, gm, 4>>, i32
        %gt = tla.arch.lanemask "gt" : i32
        tla.simt_store %out[%c3], %gt : <!tla.layout<!tla.shape<5>, !tla.stride<1>, !tla.shape<5>, RowMajor>, !tla.coord<0>, !tla.ptr<i32, gm, 4>>, i32
        %ge = tla.arch.lanemask "ge" : i32
        tla.simt_store %out[%c4], %ge : <!tla.layout<!tla.shape<5>, !tla.stride<1>, !tla.shape<5>, RowMajor>, !tla.coord<0>, !tla.ptr<i32, gm, 4>>, i32
      }) {mode = "simt", thread_block_dim = array<i64: 48, 1, 1>} : () -> ()
    }) : () -> ()
    return
  }
}

// CHECK: hivm_regbaseintrins.intrins.launch_func @lane_masks_vf_simt
// CHECK-LABEL: func.func @lane_masks_vf_simt(
// CHECK-SAME: simt_entry<48>
// eq: shifts use the lane masked to [0, 31], including lane 31.
// CHECK: %[[MASK:.*]] = arith.constant 31 : i32
// CHECK: %[[LANE:.*]] = arith.andi %{{.*}}, %[[MASK]] : i32
// CHECK: %[[ONE:.*]] = arith.constant 1 : i32
// CHECK: %[[BIT:.*]] = arith.shli %[[ONE]], %[[LANE]] : i32
// CHECK: memref.store %[[BIT]]
// lt: shifts use the lane masked to [0, 31], including lane 31.
// CHECK: %[[MASK:.*]] = arith.constant 31 : i32
// CHECK: %[[LANE:.*]] = arith.andi %{{.*}}, %[[MASK]] : i32
// CHECK: %[[ONE:.*]] = arith.constant 1 : i32
// CHECK: %[[BIT:.*]] = arith.shli %[[ONE]], %[[LANE]] : i32
// CHECK: %[[LOW:.*]] = arith.subi %[[BIT]], %[[ONE]] : i32
// CHECK: memref.store %[[LOW]]
// le: shifts use the lane masked to [0, 31], including lane 31.
// CHECK: %[[MASK:.*]] = arith.constant 31 : i32
// CHECK: %[[LANE:.*]] = arith.andi %{{.*}}, %[[MASK]] : i32
// CHECK: %[[ONE:.*]] = arith.constant 1 : i32
// CHECK: %[[BIT:.*]] = arith.shli %[[ONE]], %[[LANE]] : i32
// CHECK: %[[LOW:.*]] = arith.subi %[[BIT]], %[[ONE]] : i32
// CHECK: %[[INCLUSIVE:.*]] = arith.ori %[[BIT]], %[[LOW]] : i32
// CHECK: memref.store %[[INCLUSIVE]]
// gt: shifts use the lane masked to [0, 31], including lane 31.
// CHECK: %[[MASK:.*]] = arith.constant 31 : i32
// CHECK: %[[LANE:.*]] = arith.andi %{{.*}}, %[[MASK]] : i32
// CHECK: %[[ONE:.*]] = arith.constant 1 : i32
// CHECK: %[[BIT:.*]] = arith.shli %[[ONE]], %[[LANE]] : i32
// CHECK: %[[LOW:.*]] = arith.subi %[[BIT]], %[[ONE]] : i32
// CHECK: %[[ALL:.*]] = arith.constant -1 : i32
// CHECK: %[[INCLUSIVE:.*]] = arith.ori %[[BIT]], %[[LOW]] : i32
// CHECK: %[[RESULT:.*]] = arith.xori %[[INCLUSIVE]], %[[ALL]] : i32
// CHECK: memref.store %[[RESULT]]
// ge: shifts use the lane masked to [0, 31], including lane 31.
// CHECK: %[[MASK:.*]] = arith.constant 31 : i32
// CHECK: %[[LANE:.*]] = arith.andi %{{.*}}, %[[MASK]] : i32
// CHECK: %[[ONE:.*]] = arith.constant 1 : i32
// CHECK: %[[BIT:.*]] = arith.shli %[[ONE]], %[[LANE]] : i32
// CHECK: %[[LOW:.*]] = arith.subi %[[BIT]], %[[ONE]] : i32
// CHECK: %[[ALL:.*]] = arith.constant -1 : i32
// CHECK: %[[RESULT:.*]] = arith.xori %[[LOW]], %[[ALL]] : i32
// CHECK: memref.store %[[RESULT]]
// CHECK: return
