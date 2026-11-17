// RUN: %tla_compile %s -o - | %filecheck %s

module {
  tla.func @fill_scale_routes() {
    %0 = tla.alloc_ptr{size_bytes = 256} -> !tla.ptr<!tla.f8e8m0, l1, 512>
    %1 = tla.alloc_ptr{size_bytes = 256} -> !tla.ptr<!tla.f8e8m0, l1, 512>
    %2 = tla.make_shape -> !tla.shape<(16,2),(2,4)>
    %3 = tla.make_stride -> !tla.stride<(2,128),(1,32)>
    %4 = tla.make_shape -> !tla.shape<32,8>
    %5 = tla.make_layout %2, %3 origin %4 {layoutTag = #tla.layout_tag<zZMxScale>} : !tla.shape<(16,2),(2,4)>, !tla.stride<(2,128),(1,32)> origin !tla.shape<32,8> -> !tla.layout<!tla.shape<(16,2),(2,4)>, !tla.stride<(2,128),(1,32)>, !tla.shape<32,8>, zZMxScale>
    %6 = tla.make_coord -> !tla.coord<0,0>
    %7 = tla.make_tensor %0, %5, %6 : !tla.ptr<!tla.f8e8m0, l1, 512>, !tla.layout<!tla.shape<(16,2),(2,4)>, !tla.stride<(2,128),(1,32)>, !tla.shape<32,8>, zZMxScale>, !tla.coord<0,0> -> !tla.tensor<!tla.layout<!tla.shape<(16,2),(2,4)>, !tla.stride<(2,128),(1,32)>, !tla.shape<32,8>, zZMxScale>, !tla.coord<0,0>, !tla.ptr<!tla.f8e8m0, l1, 512>>
    %8 = tla.make_shape -> !tla.shape<(2,4),(16,2)>
    %9 = tla.make_stride -> !tla.stride<(1,32),(2,128)>
    %10 = tla.make_shape -> !tla.shape<8,32>
    %11 = tla.make_layout %8, %9 origin %10 {layoutTag = #tla.layout_tag<nNMxScale>} : !tla.shape<(2,4),(16,2)>, !tla.stride<(1,32),(2,128)> origin !tla.shape<8,32> -> !tla.layout<!tla.shape<(2,4),(16,2)>, !tla.stride<(1,32),(2,128)>, !tla.shape<8,32>, nNMxScale>
    %12 = tla.make_coord -> !tla.coord<0,0>
    %13 = tla.make_tensor %1, %11, %12 : !tla.ptr<!tla.f8e8m0, l1, 512>, !tla.layout<!tla.shape<(2,4),(16,2)>, !tla.stride<(1,32),(2,128)>, !tla.shape<8,32>, nNMxScale>, !tla.coord<0,0> -> !tla.tensor<!tla.layout<!tla.shape<(2,4),(16,2)>, !tla.stride<(1,32),(2,128)>, !tla.shape<8,32>, nNMxScale>, !tla.coord<0,0>, !tla.ptr<!tla.f8e8m0, l1, 512>>
    tla.cube {
      %c32896_i32 = arith.constant 32896 : i32
      %14 = tla.make_coord -> !tla.coord<0,0>
      %15 = tla.make_shape -> !tla.shape<32,8>
      tla.fill %7, %c32896_i32, %14, %15 : !tla.tensor<!tla.layout<!tla.shape<(16,2),(2,4)>, !tla.stride<(2,128),(1,32)>, !tla.shape<32,8>, zZMxScale>, !tla.coord<0,0>, !tla.ptr<!tla.f8e8m0, l1, 512>>, i32, <0,0>, <32,8>
      %c32382_i32 = arith.constant 32382 : i32
      %16 = tla.make_coord -> !tla.coord<0,0>
      %17 = tla.make_shape -> !tla.shape<8,32>
      tla.fill %13, %c32382_i32, %16, %17 : !tla.tensor<!tla.layout<!tla.shape<(2,4),(16,2)>, !tla.stride<(1,32),(2,128)>, !tla.shape<8,32>, nNMxScale>, !tla.coord<0,0>, !tla.ptr<!tla.f8e8m0, l1, 512>>, i32, <0,0>, <8,32>
    }
    tla.return
  }
}

// CHECK-DAG: func.func private @fill_l1_zZMxScale_uint8_t({{.*}}hivm.func_core_type = #hivm.func_core_type<AIC>

// CHECK-DAG: func.func private @fill_l1_nNMxScale_uint8_t({{.*}}hivm.func_core_type = #hivm.func_core_type<AIC>
// CHECK: call @fill_l1_zZMxScale_uint8_t
// CHECK: call @fill_l1_nNMxScale_uint8_t
// CHECK-NOT: tla.fill
