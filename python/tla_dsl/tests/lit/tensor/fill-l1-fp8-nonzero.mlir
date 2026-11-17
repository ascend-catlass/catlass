// RUN: %tla_compile %s -o - | %filecheck %s

module {
  tla.func @fill_fp8_routes() {
    %0 = tla.alloc_ptr{size_bytes = 2048} -> !tla.ptr<f8E4M3FN, l1, 512>
    %1 = tla.alloc_ptr{size_bytes = 2048} -> !tla.ptr<f8E5M2, l1, 512>
    %2 = tla.make_shape -> !tla.shape<(16,2),(32,2)>
    %3 = tla.make_stride -> !tla.stride<(32,512),(1,1024)>
    %4 = tla.make_shape -> !tla.shape<32,64>
    %5 = tla.make_layout %2, %3 origin %4 {layoutTag = #tla.layout_tag<zN>} : !tla.shape<(16,2),(32,2)>, !tla.stride<(32,512),(1,1024)> origin !tla.shape<32,64> -> !tla.layout<!tla.shape<(16,2),(32,2)>, !tla.stride<(32,512),(1,1024)>, !tla.shape<32,64>, zN>
    %6 = tla.make_coord -> !tla.coord<0,0>
    %7 = tla.make_tensor %0, %5, %6 : !tla.ptr<f8E4M3FN, l1, 512>, !tla.layout<!tla.shape<(16,2),(32,2)>, !tla.stride<(32,512),(1,1024)>, !tla.shape<32,64>, zN>, !tla.coord<0,0> -> !tla.tensor<!tla.layout<!tla.shape<(16,2),(32,2)>, !tla.stride<(32,512),(1,1024)>, !tla.shape<32,64>, zN>, !tla.coord<0,0>, !tla.ptr<f8E4M3FN, l1, 512>>
    %8 = tla.make_shape -> !tla.shape<(32,2),(16,2)>
    %9 = tla.make_stride -> !tla.stride<(1,1024),(32,512)>
    %10 = tla.make_shape -> !tla.shape<64,32>
    %11 = tla.make_layout %8, %9 origin %10 {layoutTag = #tla.layout_tag<nZ>} : !tla.shape<(32,2),(16,2)>, !tla.stride<(1,1024),(32,512)> origin !tla.shape<64,32> -> !tla.layout<!tla.shape<(32,2),(16,2)>, !tla.stride<(1,1024),(32,512)>, !tla.shape<64,32>, nZ>
    %12 = tla.make_coord -> !tla.coord<0,0>
    %13 = tla.make_tensor %1, %11, %12 : !tla.ptr<f8E5M2, l1, 512>, !tla.layout<!tla.shape<(32,2),(16,2)>, !tla.stride<(1,1024),(32,512)>, !tla.shape<64,32>, nZ>, !tla.coord<0,0> -> !tla.tensor<!tla.layout<!tla.shape<(32,2),(16,2)>, !tla.stride<(1,1024),(32,512)>, !tla.shape<64,32>, nZ>, !tla.coord<0,0>, !tla.ptr<f8E5M2, l1, 512>>
    tla.cube {
      %c14392_i32 = arith.constant 14392 : i32
      %14 = tla.make_coord -> !tla.coord<0,0>
      %15 = tla.make_shape -> !tla.shape<32,64>
      tla.fill %7, %c14392_i32, %14, %15 : !tla.tensor<!tla.layout<!tla.shape<(16,2),(32,2)>, !tla.stride<(32,512),(1,1024)>, !tla.shape<32,64>, zN>, !tla.coord<0,0>, !tla.ptr<f8E4M3FN, l1, 512>>, i32, <0,0>, <32,64>
      %c48830_i32 = arith.constant 48830 : i32
      %16 = tla.make_coord -> !tla.coord<0,0>
      %17 = tla.make_shape -> !tla.shape<64,32>
      tla.fill %13, %c48830_i32, %16, %17 : !tla.tensor<!tla.layout<!tla.shape<(32,2),(16,2)>, !tla.stride<(1,1024),(32,512)>, !tla.shape<64,32>, nZ>, !tla.coord<0,0>, !tla.ptr<f8E5M2, l1, 512>>, i32, <0,0>, <64,32>
    }
    tla.return
  }
}

// CHECK-DAG: func.func private @fill_l1_zN_fp8_e4m3fn_t({{.*}}hivm.func_core_type = #hivm.func_core_type<AIC>

// CHECK-DAG: func.func private @fill_l1_nZ_fp8_e5m2_t({{.*}}hivm.func_core_type = #hivm.func_core_type<AIC>
// CHECK: call @fill_l1_zN_fp8_e4m3fn_t
// CHECK: call @fill_l1_nZ_fp8_e5m2_t
// CHECK-NOT: tla.fill
