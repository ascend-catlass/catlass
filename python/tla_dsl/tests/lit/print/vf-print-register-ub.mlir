// RUN: %tla_compile %s -o - | %filecheck %s
// CHECK-LABEL: func.func @vf_print(
// CHECK-SAME: tla.debug_print.workspace
// CHECK: call @_mlir_ciface_tla_vf_debug_begin
// CHECK: call @vector_region_0
// CHECK-SAME: hivm.vector_function
// CHECK: call @_mlir_ciface_tla_vf_debug_end
// CHECK-LABEL: func.func private @vector_region_0
// CHECK-SAME: hivm.vector_function
// CHECK: call @_mlir_ciface_tla_vf_printf
// CHECK: call @tla_vf_print_reg_f32
// CHECK: call @tla_vf_print_ub_f32

module {
  tla.func @vf_print(%arg0: !tla.tensor<!tla.layout<!tla.shape<64>, !tla.stride<1>, !tla.shape<64>, RowMajor>, !tla.coord<0>, !tla.ptr<f32, gm, 4>>, %arg1: !tla.tensor<!tla.layout<!tla.shape<64>, !tla.stride<1>, !tla.shape<64>, RowMajor>, !tla.coord<0>, !tla.ptr<f32, gm, 4>>, %arg2: i32) {
    %0 = tla.flag "loaded" {dst_pipe = #tla.pipe<vector>, src_pipe = #tla.pipe<mte2>} -> !tla.flag
    %1 = tla.flag "done" {dst_pipe = #tla.pipe<mte3>, src_pipe = #tla.pipe<vector>} -> !tla.flag
    %2 = tla.alloc_ptr{size_bytes = 256} -> !tla.ptr<f32, ub, 256>
    %3 = tla.alloc_ptr{size_bytes = 256} -> !tla.ptr<f32, ub, 256>
    %4 = tla.make_tensor_like %2 like %arg0 layoutTag(<RowMajor>) : !tla.ptr<f32, ub, 256>, !tla.tensor<!tla.layout<!tla.shape<64>, !tla.stride<1>, !tla.shape<64>, RowMajor>, !tla.coord<0>, !tla.ptr<f32, gm, 4>> -> !tla.tensor<!tla.layout<!tla.shape<64>, !tla.stride<1>, !tla.shape<64>, RowMajor>, !tla.coord<0>, !tla.ptr<f32, ub, 256>>
    %5 = tla.make_tensor_like %3 like %arg1 layoutTag(<RowMajor>) : !tla.ptr<f32, ub, 256>, !tla.tensor<!tla.layout<!tla.shape<64>, !tla.stride<1>, !tla.shape<64>, RowMajor>, !tla.coord<0>, !tla.ptr<f32, gm, 4>> -> !tla.tensor<!tla.layout<!tla.shape<64>, !tla.stride<1>, !tla.shape<64>, RowMajor>, !tla.coord<0>, !tla.ptr<f32, ub, 256>>
    "tla.vector"() ({
      tla.copy %4, %arg0 : !tla.tensor<!tla.layout<!tla.shape<64>, !tla.stride<1>, !tla.shape<64>, RowMajor>, !tla.coord<0>, !tla.ptr<f32, ub, 256>>, !tla.tensor<!tla.layout<!tla.shape<64>, !tla.stride<1>, !tla.shape<64>, RowMajor>, !tla.coord<0>, !tla.ptr<f32, gm, 4>>
      tla.set_flag %0 : !tla.flag
      tla.wait_flag %0 : !tla.flag
      "tla.vec.func"() ({
        %result = tla.load %4 : <!tla.layout<!tla.shape<64>, !tla.stride<1>, !tla.shape<64>, RowMajor>, !tla.coord<0>, !tla.ptr<f32, ub, 256>> -> !tla.vector<64xf32>
        %cst = arith.constant 2.000000e+00 : f32
        %6 = tla.muls %result, %cst : !tla.vector<64xf32>, f32 -> !tla.vector<64xf32>
        %cst_0 = arith.constant 1.000000e+00 : f32
        %7 = tla.adds %6, %cst_0 : !tla.vector<64xf32>, f32 -> !tla.vector<64xf32>
        %cst_1 = arith.constant 2.000000e+00 : f32
        tla.debug_print format "vf_case={} scale={}"(%arg2, %cst_1) : (i32, f32)
        %c4_i64 = arith.constant 4 : i64
        tla.print_vector %7 length = %c4_i64 : !tla.vector<64xf32>, i64
        %c4_i64_2 = arith.constant 4 : i64
        tla.print_tensor %4 length = %c4_i64_2 shape = [64] : <!tla.layout<!tla.shape<64>, !tla.stride<1>, !tla.shape<64>, RowMajor>, !tla.coord<0>, !tla.ptr<f32, ub, 256>>, i64
        tla.store %5, %7 : <!tla.layout<!tla.shape<64>, !tla.stride<1>, !tla.shape<64>, RowMajor>, !tla.coord<0>, !tla.ptr<f32, ub, 256>>, !tla.vector<64xf32>
      }) {mode = "simd"} : () -> ()
      tla.set_flag %1 : !tla.flag
      tla.wait_flag %1 : !tla.flag
      tla.copy %arg1, %5 : !tla.tensor<!tla.layout<!tla.shape<64>, !tla.stride<1>, !tla.shape<64>, RowMajor>, !tla.coord<0>, !tla.ptr<f32, gm, 4>>, !tla.tensor<!tla.layout<!tla.shape<64>, !tla.stride<1>, !tla.shape<64>, RowMajor>, !tla.coord<0>, !tla.ptr<f32, ub, 256>>
      tla.pipe_barrier[<all>]
    }) : () -> ()
    tla.return
  }
}
