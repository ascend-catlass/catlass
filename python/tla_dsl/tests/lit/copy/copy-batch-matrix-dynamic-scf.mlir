// RUN: %tla_compile %s -o - | %filecheck %s
// A leading batch mode survives both if and loop tensor carriers.
// CHECK: func.func private @copy_gm_RowMajor_to_l1_zN_fp8_e4m3fn_t_batch
// CHECK-SAME: hivm.func_core_type = #hivm.func_core_type<AIC>
// CHECK-LABEL: func.func @batch_carrier
// CHECK: cf.cond_br
// CHECK: call @copy_gm_RowMajor_to_l1_zN_fp8_e4m3fn_t_batch
// CHECK-NOT: tla.tensor_desc

module {
  tla.func @batch_carrier(%arg0: i32, %arg1: i32, %arg2: i64, %arg3: i32) {
    %c4096_i64 = arith.constant 4096 : i64
    %0 = tla.inttoptr %c4096_i64 : i64 -> !tla.ptr<f8E4M3FN, gm, 1>
    %1 = arith.index_cast %arg0 : i32 to index
    %2 = arith.index_cast %arg1 : i32 to index
    %3 = tla.make_shape %1, %2 -> !tla.shape<?,?,64>
    %4 = arith.index_cast %arg2 : i64 to index
    %5 = tla.make_stride %4 -> !tla.stride<?,64,1>
    %6 = tla.make_layout %3, %5 : !tla.shape<?,?,64>, !tla.stride<?,64,1> -> !tla.layout<!tla.shape<?,?,64>, !tla.stride<?,64,1>, !tla.shape<?,?,64>, RowMajor>
    %7 = arith.subi %arg3, %arg3 : i32
    %8 = arith.index_cast %7 : i32 to index
    %9 = tla.make_coord %8 -> !tla.coord<?,0,0>
    %10 = tla.make_tensor %0, %6, %9 : !tla.ptr<f8E4M3FN, gm, 1>, !tla.layout<!tla.shape<?,?,64>, !tla.stride<?,64,1>, !tla.shape<?,?,64>, RowMajor>, !tla.coord<?,0,0> -> !tla.tensor<!tla.layout<!tla.shape<?,?,64>, !tla.stride<?,64,1>, !tla.shape<?,?,64>, RowMajor>, !tla.coord<?,0,0>, !tla.ptr<f8E4M3FN, gm, 1>>
    %c0_i32 = arith.constant 0 : i32
    %11 = arith.cmpi sgt, %arg3, %c0_i32 : i32
    %12:7 = scf.if %11 -> (!tla.tensor<!tla.layout<!tla.shape<?,?,64>, !tla.stride<?,64,1>, !tla.shape<?,?,64>, RowMajor>, !tla.coord<?,0,0>, !tla.ptr<f8E4M3FN, gm, 1>>, i32, i32, i64, i32, i32, i32) {
      %24 = arith.index_cast %arg3 : i32 to index
      %25 = tla.make_coord %24 -> !tla.coord<?,0,0>
      %26 = tla.make_tensor %0, %6, %25 : !tla.ptr<f8E4M3FN, gm, 1>, !tla.layout<!tla.shape<?,?,64>, !tla.stride<?,64,1>, !tla.shape<?,?,64>, RowMajor>, !tla.coord<?,0,0> -> !tla.tensor<!tla.layout<!tla.shape<?,?,64>, !tla.stride<?,64,1>, !tla.shape<?,?,64>, RowMajor>, !tla.coord<?,0,0>, !tla.ptr<f8E4M3FN, gm, 1>>
      scf.yield %26, %arg0, %arg1, %arg2, %arg3, %arg0, %arg1 : !tla.tensor<!tla.layout<!tla.shape<?,?,64>, !tla.stride<?,64,1>, !tla.shape<?,?,64>, RowMajor>, !tla.coord<?,0,0>, !tla.ptr<f8E4M3FN, gm, 1>>, i32, i32, i64, i32, i32, i32
    } else {
      scf.yield %10, %arg0, %arg1, %arg2, %7, %arg0, %arg1 : !tla.tensor<!tla.layout<!tla.shape<?,?,64>, !tla.stride<?,64,1>, !tla.shape<?,?,64>, RowMajor>, !tla.coord<?,0,0>, !tla.ptr<f8E4M3FN, gm, 1>>, i32, i32, i64, i32, i32, i32
    }
    %c0_i32_0 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %13:7 = scf.for %arg4 = %c0_i32_0 to %arg0 step %c1_i32 iter_args(%arg5 = %12#0, %arg6 = %12#1, %arg7 = %12#2, %arg8 = %12#3, %arg9 = %12#4, %arg10 = %12#5, %arg11 = %12#6) -> (!tla.tensor<!tla.layout<!tla.shape<?,?,64>, !tla.stride<?,64,1>, !tla.shape<?,?,64>, RowMajor>, !tla.coord<?,0,0>, !tla.ptr<f8E4M3FN, gm, 1>>, i32, i32, i64, i32, i32, i32)  : i32 {
      %24 = arith.addi %arg3, %arg4 : i32
      %25 = arith.index_cast %24 : i32 to index
      %26 = tla.make_coord %25 -> !tla.coord<?,0,0>
      %27 = tla.make_tensor %0, %6, %26 : !tla.ptr<f8E4M3FN, gm, 1>, !tla.layout<!tla.shape<?,?,64>, !tla.stride<?,64,1>, !tla.shape<?,?,64>, RowMajor>, !tla.coord<?,0,0> -> !tla.tensor<!tla.layout<!tla.shape<?,?,64>, !tla.stride<?,64,1>, !tla.shape<?,?,64>, RowMajor>, !tla.coord<?,0,0>, !tla.ptr<f8E4M3FN, gm, 1>>
      scf.yield %27, %arg0, %arg1, %arg2, %24, %arg0, %arg1 : !tla.tensor<!tla.layout<!tla.shape<?,?,64>, !tla.stride<?,64,1>, !tla.shape<?,?,64>, RowMajor>, !tla.coord<?,0,0>, !tla.ptr<f8E4M3FN, gm, 1>>, i32, i32, i64, i32, i32, i32
    }
    %14 = tla.alloc_ptr{size_bytes = 10240} -> !tla.ptr<f8E4M3FN, l1, 512>
    %15 = arith.index_cast %arg0 : i32 to index
    %16 = tla.make_shape %15 -> !tla.shape<?,(16,2),(32,2)>
    %17 = tla.make_stride -> !tla.stride<1024,(32,512),(1,5120)>
    %18 = arith.index_cast %arg0 : i32 to index
    %19 = arith.index_cast %arg1 : i32 to index
    %20 = tla.make_shape %18, %19 -> !tla.shape<?,?,64>
    %21 = tla.make_layout %16, %17 origin %20 {layoutTag = #tla.layout_tag<zN>} : !tla.shape<?,(16,2),(32,2)>, !tla.stride<1024,(32,512),(1,5120)> origin !tla.shape<?,?,64> -> !tla.layout<!tla.shape<?,(16,2),(32,2)>, !tla.stride<1024,(32,512),(1,5120)>, !tla.shape<?,?,64>, zN>
    %22 = tla.make_coord -> !tla.coord<0,0,0>
    %23 = tla.make_tensor %14, %21, %22 : !tla.ptr<f8E4M3FN, l1, 512>, !tla.layout<!tla.shape<?,(16,2),(32,2)>, !tla.stride<1024,(32,512),(1,5120)>, !tla.shape<?,?,64>, zN>, !tla.coord<0,0,0> -> !tla.tensor<!tla.layout<!tla.shape<?,(16,2),(32,2)>, !tla.stride<1024,(32,512),(1,5120)>, !tla.shape<?,?,64>, zN>, !tla.coord<0,0,0>, !tla.ptr<f8E4M3FN, l1, 512>>
    tla.cube {
      tla.copy %23, %13#0 : !tla.tensor<!tla.layout<!tla.shape<?,(16,2),(32,2)>, !tla.stride<1024,(32,512),(1,5120)>, !tla.shape<?,?,64>, zN>, !tla.coord<0,0,0>, !tla.ptr<f8E4M3FN, l1, 512>>, !tla.tensor<!tla.layout<!tla.shape<?,?,64>, !tla.stride<?,64,1>, !tla.shape<?,?,64>, RowMajor>, !tla.coord<?,0,0>, !tla.ptr<f8E4M3FN, gm, 1>>
    }
    tla.return
  }
}
