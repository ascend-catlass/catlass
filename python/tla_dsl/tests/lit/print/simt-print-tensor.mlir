// RUN: %tla_compile %s -o - | %filecheck %s
// CHECK-DAG: func.func private @_mlir_ciface_tla_simt_print_gm(i64, !llvm.ptr<1>, i64, i32, i32, i32, i32, i32)
// CHECK-DAG: func.func private @_mlir_ciface_tla_simt_print_ub(i64, !llvm.ptr<6>, i64, i32, i32, i32, i32, i32)
// CHECK-LABEL: func.func @simt_print(
// CHECK-SAME: tla.simt_print.workspace
// CHECK-SAME: tla.debug_print.workspace
// CHECK: launch_func @simt_print_vf_simt
// CHECK-LABEL: func.func @simt_print_vf_simt(
// CHECK: cf.cond_br
// CHECK: call @_mlir_ciface_tla_simt_print_ub
// CHECK: call @_mlir_ciface_tla_simt_print_gm
// CHECK: call @_mlir_ciface_tla_simt_printf
module {
  tla.func @simt_print(%arg0: !tla.tensor<!tla.layout<!tla.shape<64>, !tla.stride<1>, !tla.shape<64>, RowMajor>, !tla.coord<0>, !tla.ptr<f32, gm, 4>>, %arg1: !tla.tensor<!tla.layout<!tla.shape<64>, !tla.stride<1>, !tla.shape<64>, RowMajor>, !tla.coord<0>, !tla.ptr<f32, gm, 4>>, %arg2: i32) {
    %0 = tla.alloc_ptr{size_bytes = 256} -> !tla.ptr<f32, ub, 256>
    %1 = tla.make_tensor_like %0 like %arg0 layoutTag(<RowMajor>) : !tla.ptr<f32, ub, 256>, !tla.tensor<!tla.layout<!tla.shape<64>, !tla.stride<1>, !tla.shape<64>, RowMajor>, !tla.coord<0>, !tla.ptr<f32, gm, 4>> -> !tla.tensor<!tla.layout<!tla.shape<64>, !tla.stride<1>, !tla.shape<64>, RowMajor>, !tla.coord<0>, !tla.ptr<f32, ub, 256>>
    "tla.vector"() ({
      tla.debug_print format "outer={}"(%arg2) : (i32)
      "tla.vec.func"() ({
        tla.debug_print format "simd={}"(%arg2) : (i32)
      }) {mode = "simd"} : () -> ()
      "tla.vec.func"() ({
        %x, %y, %z = tla.arch.thread_idx -> i32, i32, i32
        %2 = arith.index_cast %x : i32 to index
        %3 = tla.simt_load %arg0[%2] : <!tla.layout<!tla.shape<64>, !tla.stride<1>, !tla.shape<64>, RowMajor>, !tla.coord<0>, !tla.ptr<f32, gm, 4>> -> f32
        %cst = arith.constant 2.000000e+00 : f32
        %4 = tla.simt_mul %3, %cst : f32
        %5 = arith.index_cast %x : i32 to index
        tla.simt_store %1[%5], %4 : <!tla.layout<!tla.shape<64>, !tla.stride<1>, !tla.shape<64>, RowMajor>, !tla.coord<0>, !tla.ptr<f32, ub, 256>>, f32
        tla.arch.sync_threads
        %c0_i32 = arith.constant 0 : i32
        %6 = tla.simt_cmp "eq" %x, %c0_i32 : i32
        scf.if %6 {
          %12 = arith.extsi %arg2 : i32 to i64
          tla.print_tensor %1 length = %12 shape = [64] {dynamic_execution} : <!tla.layout<!tla.shape<64>, !tla.stride<1>, !tla.shape<64>, RowMajor>, !tla.coord<0>, !tla.ptr<f32, ub, 256>>, i64
          %13 = arith.extsi %arg2 : i32 to i64
          tla.print_tensor %arg0 length = %13 shape = [64] {dynamic_execution} : <!tla.layout<!tla.shape<64>, !tla.stride<1>, !tla.shape<64>, RowMajor>, !tla.coord<0>, !tla.ptr<f32, gm, 4>>, i64
        }
        %c4_i32 = arith.constant 4 : i32
        %7 = tla.simt_cmp "lt" %x, %c4_i32 : i32
        scf.if %7 {
          %12 = arith.index_cast %x : i32 to index
          %13 = tla.simt_load %1[%12] : <!tla.layout<!tla.shape<64>, !tla.stride<1>, !tla.shape<64>, RowMajor>, !tla.coord<0>, !tla.ptr<f32, ub, 256>> -> f32
          tla.debug_print format "tid={} x={}"(%x, %13) : (i32, f32)
        }
        %8 = arith.index_cast %x : i32 to index
        %9 = tla.simt_load %1[%8] : <!tla.layout<!tla.shape<64>, !tla.stride<1>, !tla.shape<64>, RowMajor>, !tla.coord<0>, !tla.ptr<f32, ub, 256>> -> f32
        %cst_0 = arith.constant 1.000000e+00 : f32
        %10 = tla.simt_add %9, %cst_0 : f32
        %11 = arith.index_cast %x : i32 to index
        tla.simt_store %arg1[%11], %10 : <!tla.layout<!tla.shape<64>, !tla.stride<1>, !tla.shape<64>, RowMajor>, !tla.coord<0>, !tla.ptr<f32, gm, 4>>, f32
      }) {mode = "simt", thread_block_dim = array<i64: 64, 1, 1>} : () -> ()
    }) : () -> ()
    tla.return
  }
}
