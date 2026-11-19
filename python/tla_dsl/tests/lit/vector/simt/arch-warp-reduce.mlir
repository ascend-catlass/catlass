// RUN: %tla_compile %s --mlir-print-ir-after-all -o %t.cse 2>&1 | %filecheck %s --check-prefix=CONVERGENCE
// RUN: %tla_compile %s --mlir-print-ir-after=tla-vector-region-simt -o %t 2>&1 | %filecheck %s --implicit-check-not=tla.arch.warp_reduce
module {
  func.func @i32_s32(%mem: memref<3xi32, #hivm.address_space<gm>>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index
    %out = tla.tensor_desc %mem shape [%c3, %c1, %c1, %c1] stride [%c1, %c1, %c1, %c1] origin_shape [%c3, %c1] coord [%c0, %c0] : memref<3xi32, #hivm.address_space<gm>> -> !tla.tensor<!tla.layout<!tla.shape<3>, !tla.stride<1>, !tla.shape<3>, RowMajor>, !tla.coord<0>, !tla.ptr<i32, gm, 4>>
    "tla.vector"() ({
      "tla.vec.func"() ({
        %v = tla.simt_load %out[%c0] : <!tla.layout<!tla.shape<3>, !tla.stride<1>, !tla.shape<3>, RowMajor>, !tla.coord<0>, !tla.ptr<i32, gm, 4>> -> i32
        %add = tla.arch.warp_reduce "add" %v : i32
        tla.simt_store %out[%c0], %add : <!tla.layout<!tla.shape<3>, !tla.stride<1>, !tla.shape<3>, RowMajor>, !tla.coord<0>, !tla.ptr<i32, gm, 4>>, i32
        %max = tla.arch.warp_reduce "max" %v : i32
        tla.simt_store %out[%c1], %max : <!tla.layout<!tla.shape<3>, !tla.stride<1>, !tla.shape<3>, RowMajor>, !tla.coord<0>, !tla.ptr<i32, gm, 4>>, i32
        %min = tla.arch.warp_reduce "min" %v : i32
        tla.simt_store %out[%c2], %min : <!tla.layout<!tla.shape<3>, !tla.stride<1>, !tla.shape<3>, RowMajor>, !tla.coord<0>, !tla.ptr<i32, gm, 4>>, i32
        %tx, %ty, %tz = tla.arch.thread_idx -> i32, i32, i32
        %half = arith.constant 16 : i32
        %active = arith.cmpi slt, %tx, %half : i32
        scf.if %active {
          %partial = tla.arch.warp_reduce "add" %v : i32
          tla.simt_store %out[%c1], %partial : <!tla.layout<!tla.shape<3>, !tla.stride<1>, !tla.shape<3>, RowMajor>, !tla.coord<0>, !tla.ptr<i32, gm, 4>>, i32
        }
        scf.for %i = %c0 to %c2 step %c1 {
          %loop = tla.arch.warp_reduce "max" %v : i32
          tla.simt_store %out[%c2], %loop : <!tla.layout<!tla.shape<3>, !tla.stride<1>, !tla.shape<3>, RowMajor>, !tla.coord<0>, !tla.ptr<i32, gm, 4>>, i32
        }
      }) {mode = "simt", thread_block_dim = array<i64: 48, 1, 1>} : () -> ()
    }) : () -> ()
    return
  }
  func.func @i32_u32(%mem: memref<3xi32, #hivm.address_space<gm>>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index
    %out = tla.tensor_desc %mem shape [%c3, %c1, %c1, %c1] stride [%c1, %c1, %c1, %c1] origin_shape [%c3, %c1] coord [%c0, %c0] : memref<3xi32, #hivm.address_space<gm>> -> !tla.tensor<!tla.layout<!tla.shape<3>, !tla.stride<1>, !tla.shape<3>, RowMajor>, !tla.coord<0>, !tla.ptr<i32, gm, 4>>
    "tla.vector"() ({
      "tla.vec.func"() ({
        %v = tla.simt_load %out[%c0] : <!tla.layout<!tla.shape<3>, !tla.stride<1>, !tla.shape<3>, RowMajor>, !tla.coord<0>, !tla.ptr<i32, gm, 4>> -> i32
        %add = tla.arch.warp_reduce "add" %v {isUnsigned} : i32
        tla.simt_store %out[%c0], %add : <!tla.layout<!tla.shape<3>, !tla.stride<1>, !tla.shape<3>, RowMajor>, !tla.coord<0>, !tla.ptr<i32, gm, 4>>, i32
        %max = tla.arch.warp_reduce "max" %v {isUnsigned} : i32
        tla.simt_store %out[%c1], %max : <!tla.layout<!tla.shape<3>, !tla.stride<1>, !tla.shape<3>, RowMajor>, !tla.coord<0>, !tla.ptr<i32, gm, 4>>, i32
        %min = tla.arch.warp_reduce "min" %v {isUnsigned} : i32
        tla.simt_store %out[%c2], %min : <!tla.layout<!tla.shape<3>, !tla.stride<1>, !tla.shape<3>, RowMajor>, !tla.coord<0>, !tla.ptr<i32, gm, 4>>, i32
      }) {mode = "simt", thread_block_dim = array<i64: 48, 1, 1>} : () -> ()
    }) : () -> ()
    return
  }
  func.func @f16_f16(%mem: memref<3xf16, #hivm.address_space<gm>>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index
    %out = tla.tensor_desc %mem shape [%c3, %c1, %c1, %c1] stride [%c1, %c1, %c1, %c1] origin_shape [%c3, %c1] coord [%c0, %c0] : memref<3xf16, #hivm.address_space<gm>> -> !tla.tensor<!tla.layout<!tla.shape<3>, !tla.stride<1>, !tla.shape<3>, RowMajor>, !tla.coord<0>, !tla.ptr<f16, gm, 4>>
    "tla.vector"() ({
      "tla.vec.func"() ({
        %v = tla.simt_load %out[%c0] : <!tla.layout<!tla.shape<3>, !tla.stride<1>, !tla.shape<3>, RowMajor>, !tla.coord<0>, !tla.ptr<f16, gm, 4>> -> f16
        %add = tla.arch.warp_reduce "add" %v : f16
        tla.simt_store %out[%c0], %add : <!tla.layout<!tla.shape<3>, !tla.stride<1>, !tla.shape<3>, RowMajor>, !tla.coord<0>, !tla.ptr<f16, gm, 4>>, f16
        %max = tla.arch.warp_reduce "max" %v : f16
        tla.simt_store %out[%c1], %max : <!tla.layout<!tla.shape<3>, !tla.stride<1>, !tla.shape<3>, RowMajor>, !tla.coord<0>, !tla.ptr<f16, gm, 4>>, f16
        %min = tla.arch.warp_reduce "min" %v : f16
        tla.simt_store %out[%c2], %min : <!tla.layout<!tla.shape<3>, !tla.stride<1>, !tla.shape<3>, RowMajor>, !tla.coord<0>, !tla.ptr<f16, gm, 4>>, f16
      }) {mode = "simt", thread_block_dim = array<i64: 48, 1, 1>} : () -> ()
    }) : () -> ()
    return
  }
  func.func @f32_f32(%mem: memref<3xf32, #hivm.address_space<gm>>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index
    %out = tla.tensor_desc %mem shape [%c3, %c1, %c1, %c1] stride [%c1, %c1, %c1, %c1] origin_shape [%c3, %c1] coord [%c0, %c0] : memref<3xf32, #hivm.address_space<gm>> -> !tla.tensor<!tla.layout<!tla.shape<3>, !tla.stride<1>, !tla.shape<3>, RowMajor>, !tla.coord<0>, !tla.ptr<f32, gm, 4>>
    "tla.vector"() ({
      "tla.vec.func"() ({
        %v = tla.simt_load %out[%c0] : <!tla.layout<!tla.shape<3>, !tla.stride<1>, !tla.shape<3>, RowMajor>, !tla.coord<0>, !tla.ptr<f32, gm, 4>> -> f32
        %add = tla.arch.warp_reduce "add" %v : f32
        tla.simt_store %out[%c0], %add : <!tla.layout<!tla.shape<3>, !tla.stride<1>, !tla.shape<3>, RowMajor>, !tla.coord<0>, !tla.ptr<f32, gm, 4>>, f32
        %max = tla.arch.warp_reduce "max" %v : f32
        tla.simt_store %out[%c1], %max : <!tla.layout<!tla.shape<3>, !tla.stride<1>, !tla.shape<3>, RowMajor>, !tla.coord<0>, !tla.ptr<f32, gm, 4>>, f32
        %min = tla.arch.warp_reduce "min" %v : f32
        tla.simt_store %out[%c2], %min : <!tla.layout<!tla.shape<3>, !tla.stride<1>, !tla.shape<3>, RowMajor>, !tla.coord<0>, !tla.ptr<f32, gm, 4>>, f32
      }) {mode = "simt", thread_block_dim = array<i64: 48, 1, 1>} : () -> ()
    }) : () -> ()
    return
  }
  func.func @i8_s32(%mem: memref<1xi8, #hivm.address_space<gm>>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %out = tla.tensor_desc %mem shape [%c1, %c1, %c1, %c1] stride [%c1, %c1, %c1, %c1] origin_shape [%c1, %c1] coord [%c0, %c0] : memref<1xi8, #hivm.address_space<gm>> -> !tla.tensor<!tla.layout<!tla.shape<1>, !tla.stride<1>, !tla.shape<1>, RowMajor>, !tla.coord<0>, !tla.ptr<i8, gm, 4>>
    "tla.vector"() ({
      "tla.vec.func"() ({
        %v = tla.simt_load %out[%c0] : <!tla.layout<!tla.shape<1>, !tla.stride<1>, !tla.shape<1>, RowMajor>, !tla.coord<0>, !tla.ptr<i8, gm, 4>> -> i8
        %add = tla.arch.warp_reduce "add" %v : i8
        tla.simt_store %out[%c0], %add : <!tla.layout<!tla.shape<1>, !tla.stride<1>, !tla.shape<1>, RowMajor>, !tla.coord<0>, !tla.ptr<i8, gm, 4>>, i8
      }) {mode = "simt", thread_block_dim = array<i64: 48, 1, 1>} : () -> ()
    }) : () -> ()
    return
  }
  func.func @i8_u32(%mem: memref<1xi8, #hivm.address_space<gm>>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %out = tla.tensor_desc %mem shape [%c1, %c1, %c1, %c1] stride [%c1, %c1, %c1, %c1] origin_shape [%c1, %c1] coord [%c0, %c0] : memref<1xi8, #hivm.address_space<gm>> -> !tla.tensor<!tla.layout<!tla.shape<1>, !tla.stride<1>, !tla.shape<1>, RowMajor>, !tla.coord<0>, !tla.ptr<i8, gm, 4>>
    "tla.vector"() ({
      "tla.vec.func"() ({
        %v = tla.simt_load %out[%c0] : <!tla.layout<!tla.shape<1>, !tla.stride<1>, !tla.shape<1>, RowMajor>, !tla.coord<0>, !tla.ptr<i8, gm, 4>> -> i8
        %max = tla.arch.warp_reduce "max" %v {isUnsigned} : i8
        tla.simt_store %out[%c0], %max : <!tla.layout<!tla.shape<1>, !tla.stride<1>, !tla.shape<1>, RowMajor>, !tla.coord<0>, !tla.ptr<i8, gm, 4>>, i8
      }) {mode = "simt", thread_block_dim = array<i64: 48, 1, 1>} : () -> ()
    }) : () -> ()
    return
  }
  func.func @i16_s32(%mem: memref<1xi16, #hivm.address_space<gm>>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %out = tla.tensor_desc %mem shape [%c1, %c1, %c1, %c1] stride [%c1, %c1, %c1, %c1] origin_shape [%c1, %c1] coord [%c0, %c0] : memref<1xi16, #hivm.address_space<gm>> -> !tla.tensor<!tla.layout<!tla.shape<1>, !tla.stride<1>, !tla.shape<1>, RowMajor>, !tla.coord<0>, !tla.ptr<i16, gm, 4>>
    "tla.vector"() ({
      "tla.vec.func"() ({
        %v = tla.simt_load %out[%c0] : <!tla.layout<!tla.shape<1>, !tla.stride<1>, !tla.shape<1>, RowMajor>, !tla.coord<0>, !tla.ptr<i16, gm, 4>> -> i16
        %min = tla.arch.warp_reduce "min" %v : i16
        tla.simt_store %out[%c0], %min : <!tla.layout<!tla.shape<1>, !tla.stride<1>, !tla.shape<1>, RowMajor>, !tla.coord<0>, !tla.ptr<i16, gm, 4>>, i16
      }) {mode = "simt", thread_block_dim = array<i64: 48, 1, 1>} : () -> ()
    }) : () -> ()
    return
  }
  func.func @i16_u32(%mem: memref<1xi16, #hivm.address_space<gm>>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %out = tla.tensor_desc %mem shape [%c1, %c1, %c1, %c1] stride [%c1, %c1, %c1, %c1] origin_shape [%c1, %c1] coord [%c0, %c0] : memref<1xi16, #hivm.address_space<gm>> -> !tla.tensor<!tla.layout<!tla.shape<1>, !tla.stride<1>, !tla.shape<1>, RowMajor>, !tla.coord<0>, !tla.ptr<i16, gm, 4>>
    "tla.vector"() ({
      "tla.vec.func"() ({
        %v = tla.simt_load %out[%c0] : <!tla.layout<!tla.shape<1>, !tla.stride<1>, !tla.shape<1>, RowMajor>, !tla.coord<0>, !tla.ptr<i16, gm, 4>> -> i16
        %add = tla.arch.warp_reduce "add" %v {isUnsigned} : i16
        tla.simt_store %out[%c0], %add : <!tla.layout<!tla.shape<1>, !tla.stride<1>, !tla.shape<1>, RowMajor>, !tla.coord<0>, !tla.ptr<i16, gm, 4>>, i16
      }) {mode = "simt", thread_block_dim = array<i64: 48, 1, 1>} : () -> ()
    }) : () -> ()
    return
  }

}

// CHECK-LABEL: func.func @i32_s32_vf_simt{{[0-9]*}}(
// CHECK: hivm_regbaseintrins.intr.hivm.redux.add.s32
// CHECK: hivm_regbaseintrins.intr.hivm.redux.max.s32
// CHECK: hivm_regbaseintrins.intr.hivm.redux.min.s32
// CHECK-LABEL: func.func @i32_u32_vf_simt{{[0-9]*}}(
// CHECK: hivm_regbaseintrins.intr.hivm.redux.add.u32
// CHECK: hivm_regbaseintrins.intr.hivm.redux.max.u32
// CHECK: hivm_regbaseintrins.intr.hivm.redux.min.u32
// CHECK-LABEL: func.func @f16_f16_vf_simt{{[0-9]*}}(
// CHECK: hivm_regbaseintrins.intr.hivm.redux.add.f16
// CHECK: hivm_regbaseintrins.intr.hivm.redux.max.f16
// CHECK: hivm_regbaseintrins.intr.hivm.redux.min.f16
// CHECK-LABEL: func.func @f32_f32_vf_simt{{[0-9]*}}(
// CHECK: hivm_regbaseintrins.intr.hivm.redux.add.f32
// CHECK: hivm_regbaseintrins.intr.hivm.redux.max.f32
// CHECK: hivm_regbaseintrins.intr.hivm.redux.min.f32
// CHECK-LABEL: func.func @i8_s32_vf_simt{{[0-9]*}}(
// CHECK: arith.extsi
// CHECK: hivm_regbaseintrins.intr.hivm.redux.add.s32
// CHECK: arith.trunci
// CHECK-LABEL: func.func @i8_u32_vf_simt{{[0-9]*}}(
// CHECK: arith.extui
// CHECK: hivm_regbaseintrins.intr.hivm.redux.max.u32
// CHECK: arith.trunci
// CHECK-LABEL: func.func @i16_s32_vf_simt{{[0-9]*}}(
// CHECK: arith.extsi
// CHECK: hivm_regbaseintrins.intr.hivm.redux.min.s32
// CHECK: arith.trunci
// CHECK-LABEL: func.func @i16_u32_vf_simt{{[0-9]*}}(
// CHECK: arith.extui
// CHECK: hivm_regbaseintrins.intr.hivm.redux.add.u32
// CHECK: arith.trunci
// CONVERGENCE-LABEL: IR Dump After CSE (cse)
// CONVERGENCE: func.func @i32_s32_vf_simt
// CONVERGENCE: hivm_regbaseintrins.intr.hivm.redux.add.s32
// CONVERGENCE: scf.if
// CONVERGENCE: hivm_regbaseintrins.intr.hivm.redux.add.s32
// CONVERGENCE: memref.store
// CONVERGENCE: scf.for
// CONVERGENCE: hivm_regbaseintrins.intr.hivm.redux.max.s32
