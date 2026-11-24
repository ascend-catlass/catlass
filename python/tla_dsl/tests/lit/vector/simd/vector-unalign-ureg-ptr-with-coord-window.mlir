// RUN: %tla_compile %s --mlir-print-ir-after=tla-unalign-ureg -o %t 2>&1 | %filecheck %s

// UnAlign UPDATE store tails must be flushed before a later read of the same
// UB allocation. tla-lower-ptr expands a view start into chained i64 address
// additions: the store window is addi(addi(ptrtoint(arena), slot), offset) and
// rebuilding the read window through t.ptr_with_coord adds one more addi. The
// conflict analysis must trace through every addi level; otherwise it sees two
// different roots, defers store_unalign_post to after the loop, and the reader
// sees the pre-store bytes.
//
// The first function pins that ordering for the ptr_with_coord spelling. The
// second keeps the original optimisation alive: reading from an independent UB
// allocation is not a conflict, so its POST still tips after the loop.

module {
  tla.func @ptr_with_coord_window(%arg0: !tla.tensor<!tla.layout<!tla.shape<64>, !tla.stride<1>, !tla.shape<64>, RowMajor>, !tla.coord<0>, !tla.ptr<f32, gm, 4>>) {
    %0 = tla.make_shape -> !tla.shape<64>
    %1 = tla.make_stride -> !tla.stride<1>
    %2 = tla.make_shape -> !tla.shape<64>
    %3 = tla.make_layout %0, %1 origin %2 : !tla.shape<64>, !tla.stride<1> origin !tla.shape<64> -> !tla.layout<!tla.shape<64>, !tla.stride<1>, !tla.shape<64>, RowMajor>
    %4 = tla.make_shape -> !tla.shape<128>
    %5 = tla.make_stride -> !tla.stride<1>
    %6 = tla.make_shape -> !tla.shape<128>
    %7 = tla.make_layout %4, %5 origin %6 : !tla.shape<128>, !tla.stride<1> origin !tla.shape<128> -> !tla.layout<!tla.shape<128>, !tla.stride<1>, !tla.shape<128>, RowMajor>
    %8 = tla.alloc_ptr{size_bytes = 512} -> !tla.ptr<f32, ub, 256>
    %9 = tla.alloc_ptr{size_bytes = 512} -> !tla.ptr<f32, ub, 256>
    %10 = tla.alloc_ptr{size_bytes = 256} -> !tla.ptr<f32, ub, 256>
    %11 = tla.make_coord -> !tla.coord<0>
    %12 = tla.make_tensor %9, %7, %11 : !tla.ptr<f32, ub, 256>, !tla.layout<!tla.shape<128>, !tla.stride<1>, !tla.shape<128>, RowMajor>, !tla.coord<0> -> !tla.tensor<!tla.layout<!tla.shape<128>, !tla.stride<1>, !tla.shape<128>, RowMajor>, !tla.coord<0>, !tla.ptr<f32, ub, 256>>
    %13 = tla.make_shape -> !tla.shape<1>
    %14 = tla.make_coord -> !tla.coord<1>
    %15 = tla.make_coord -> !tla.coord<1>
    %16 = tla.tile_view %12, %13, %15 : !tla.tensor<!tla.layout<!tla.shape<128>, !tla.stride<1>, !tla.shape<128>, RowMajor>, !tla.coord<0>, !tla.ptr<f32, ub, 256>>, !tla.shape<1>, !tla.coord<1> -> !tla.tensor<!tla.layout<!tla.shape<1>, !tla.stride<1>, !tla.shape<1>, RowMajor>, !tla.coord<1>, !tla.ptr<f32, ub, 256>>
    %17 = tla.tensor_ptr_with_coord %16 : !tla.tensor<!tla.layout<!tla.shape<1>, !tla.stride<1>, !tla.shape<1>, RowMajor>, !tla.coord<1>, !tla.ptr<f32, ub, 256>> -> !tla.ptr<f32, ub, 256>
    %18 = tla.make_coord -> !tla.coord<0>
    %19 = tla.make_tensor %17, %3, %18 : !tla.ptr<f32, ub, 256>, !tla.layout<!tla.shape<64>, !tla.stride<1>, !tla.shape<64>, RowMajor>, !tla.coord<0> -> !tla.tensor<!tla.layout<!tla.shape<64>, !tla.stride<1>, !tla.shape<64>, RowMajor>, !tla.coord<0>, !tla.ptr<f32, ub, 256>>
    %20 = tla.make_shape -> !tla.shape<1>
    %21 = tla.make_coord -> !tla.coord<1>
    %22 = tla.make_coord -> !tla.coord<1>
    %23 = tla.tile_view %19, %20, %22 : !tla.tensor<!tla.layout<!tla.shape<64>, !tla.stride<1>, !tla.shape<64>, RowMajor>, !tla.coord<0>, !tla.ptr<f32, ub, 256>>, !tla.shape<1>, !tla.coord<1> -> !tla.tensor<!tla.layout<!tla.shape<1>, !tla.stride<1>, !tla.shape<1>, RowMajor>, !tla.coord<1>, !tla.ptr<f32, ub, 256>>
    %24 = tla.tensor_ptr_with_coord %23 : !tla.tensor<!tla.layout<!tla.shape<1>, !tla.stride<1>, !tla.shape<1>, RowMajor>, !tla.coord<1>, !tla.ptr<f32, ub, 256>> -> !tla.ptr<f32, ub, 256>
    %25 = tla.make_coord -> !tla.coord<0>
    %26 = tla.make_tensor %24, %3, %25 : !tla.ptr<f32, ub, 256>, !tla.layout<!tla.shape<64>, !tla.stride<1>, !tla.shape<64>, RowMajor>, !tla.coord<0> -> !tla.tensor<!tla.layout<!tla.shape<64>, !tla.stride<1>, !tla.shape<64>, RowMajor>, !tla.coord<0>, !tla.ptr<f32, ub, 256>>
    %27 = tla.make_coord -> !tla.coord<0>
    %28 = tla.make_tensor %10, %3, %27 : !tla.ptr<f32, ub, 256>, !tla.layout<!tla.shape<64>, !tla.stride<1>, !tla.shape<64>, RowMajor>, !tla.coord<0> -> !tla.tensor<!tla.layout<!tla.shape<64>, !tla.stride<1>, !tla.shape<64>, RowMajor>, !tla.coord<0>, !tla.ptr<f32, ub, 256>>
    "tla.vector"() ({
      "tla.vec.func"() ({
        %cst = arith.constant 1.100000e+01 : f32
        %29 = tla.full %cst : f32 -> !tla.vector<64xf32>
        %c0_i32 = arith.constant 0 : i32
        %c1_i32 = arith.constant 1 : i32
        %c1_i32_0 = arith.constant 1 : i32
        %30 = scf.for %arg1 = %c0_i32 to %c1_i32 step %c1_i32_0 iter_args(%arg2 = %19) -> (!tla.tensor<!tla.layout<!tla.shape<64>, !tla.stride<1>, !tla.shape<64>, RowMajor>, !tla.coord<0>, !tla.ptr<f32, ub, 256>>)  : i32 {
          %c64_i32_1 = arith.constant 64 : i32
          %next_ptr_2, %ureg_out_3 = tla.store_unalign_update %arg2, %29 post_update_stride %c64_i32_1 : <!tla.layout<!tla.shape<64>, !tla.stride<1>, !tla.shape<64>, RowMajor>, !tla.coord<0>, !tla.ptr<f32, ub, 256>>, !tla.vector<64xf32> post_update_stride i32 -> !tla.ptr<f32, ub, 256>, !tla.ureg
          %37 = tla.make_shape -> !tla.shape<64>
          %38 = tla.make_stride -> !tla.stride<1>
          %39 = tla.make_shape -> !tla.shape<64>
          %40 = tla.make_layout %37, %38 origin %39 : !tla.shape<64>, !tla.stride<1> origin !tla.shape<64> -> !tla.layout<!tla.shape<64>, !tla.stride<1>, !tla.shape<64>, RowMajor>
          %41 = tla.make_coord -> !tla.coord<0>
          %42 = tla.make_tensor %next_ptr_2, %40, %41 : !tla.ptr<f32, ub, 256>, !tla.layout<!tla.shape<64>, !tla.stride<1>, !tla.shape<64>, RowMajor>, !tla.coord<0> -> !tla.tensor<!tla.layout<!tla.shape<64>, !tla.stride<1>, !tla.shape<64>, RowMajor>, !tla.coord<0>, !tla.ptr<f32, ub, 256>>
          tla.local_mem_bar 1
          %result = tla.load %26 {unaligned_ub_access} : <!tla.layout<!tla.shape<64>, !tla.stride<1>, !tla.shape<64>, RowMajor>, !tla.coord<0>, !tla.ptr<f32, ub, 256>> -> !tla.vector<64xf32>
          tla.store %28, %result : <!tla.layout<!tla.shape<64>, !tla.stride<1>, !tla.shape<64>, RowMajor>, !tla.coord<0>, !tla.ptr<f32, ub, 256>>, !tla.vector<64xf32>
          scf.yield %42 : !tla.tensor<!tla.layout<!tla.shape<64>, !tla.stride<1>, !tla.shape<64>, RowMajor>, !tla.coord<0>, !tla.ptr<f32, ub, 256>>
        }
        %c64_i32 = arith.constant 64 : i32
        %next_ptr, %ureg_out = tla.store_unalign_update %30, %29 post_update_stride %c64_i32 : <!tla.layout<!tla.shape<64>, !tla.stride<1>, !tla.shape<64>, RowMajor>, !tla.coord<0>, !tla.ptr<f32, ub, 256>>, !tla.vector<64xf32> post_update_stride i32 -> !tla.ptr<f32, ub, 256>, !tla.ureg
        %31 = tla.make_shape -> !tla.shape<64>
        %32 = tla.make_stride -> !tla.stride<1>
        %33 = tla.make_shape -> !tla.shape<64>
        %34 = tla.make_layout %31, %32 origin %33 : !tla.shape<64>, !tla.stride<1> origin !tla.shape<64> -> !tla.layout<!tla.shape<64>, !tla.stride<1>, !tla.shape<64>, RowMajor>
        %35 = tla.make_coord -> !tla.coord<0>
        %36 = tla.make_tensor %next_ptr, %34, %35 : !tla.ptr<f32, ub, 256>, !tla.layout<!tla.shape<64>, !tla.stride<1>, !tla.shape<64>, RowMajor>, !tla.coord<0> -> !tla.tensor<!tla.layout<!tla.shape<64>, !tla.stride<1>, !tla.shape<64>, RowMajor>, !tla.coord<0>, !tla.ptr<f32, ub, 256>>
      }) {mode = "simd"} : () -> ()
      tla.copy %28, %arg0 : !tla.tensor<!tla.layout<!tla.shape<64>, !tla.stride<1>, !tla.shape<64>, RowMajor>, !tla.coord<0>, !tla.ptr<f32, ub, 256>>, !tla.tensor<!tla.layout<!tla.shape<64>, !tla.stride<1>, !tla.shape<64>, RowMajor>, !tla.coord<0>, !tla.ptr<f32, gm, 4>>
    }) : () -> ()
    tla.return
  }
  tla.func @separate_windows(%arg0: !tla.tensor<!tla.layout<!tla.shape<64>, !tla.stride<1>, !tla.shape<64>, RowMajor>, !tla.coord<0>, !tla.ptr<f32, gm, 4>>) {
    %0 = tla.make_shape -> !tla.shape<64>
    %1 = tla.make_stride -> !tla.stride<1>
    %2 = tla.make_shape -> !tla.shape<64>
    %3 = tla.make_layout %0, %1 origin %2 : !tla.shape<64>, !tla.stride<1> origin !tla.shape<64> -> !tla.layout<!tla.shape<64>, !tla.stride<1>, !tla.shape<64>, RowMajor>
    %4 = tla.make_shape -> !tla.shape<128>
    %5 = tla.make_stride -> !tla.stride<1>
    %6 = tla.make_shape -> !tla.shape<128>
    %7 = tla.make_layout %4, %5 origin %6 : !tla.shape<128>, !tla.stride<1> origin !tla.shape<128> -> !tla.layout<!tla.shape<128>, !tla.stride<1>, !tla.shape<128>, RowMajor>
    %8 = tla.alloc_ptr{size_bytes = 512} -> !tla.ptr<f32, ub, 256>
    %9 = tla.alloc_ptr{size_bytes = 512} -> !tla.ptr<f32, ub, 256>
    %10 = tla.alloc_ptr{size_bytes = 512} -> !tla.ptr<f32, ub, 256>
    %11 = tla.alloc_ptr{size_bytes = 256} -> !tla.ptr<f32, ub, 256>
    %12 = tla.make_coord -> !tla.coord<0>
    %13 = tla.make_tensor %9, %7, %12 : !tla.ptr<f32, ub, 256>, !tla.layout<!tla.shape<128>, !tla.stride<1>, !tla.shape<128>, RowMajor>, !tla.coord<0> -> !tla.tensor<!tla.layout<!tla.shape<128>, !tla.stride<1>, !tla.shape<128>, RowMajor>, !tla.coord<0>, !tla.ptr<f32, ub, 256>>
    %14 = tla.make_shape -> !tla.shape<1>
    %15 = tla.make_coord -> !tla.coord<1>
    %16 = tla.make_coord -> !tla.coord<1>
    %17 = tla.tile_view %13, %14, %16 : !tla.tensor<!tla.layout<!tla.shape<128>, !tla.stride<1>, !tla.shape<128>, RowMajor>, !tla.coord<0>, !tla.ptr<f32, ub, 256>>, !tla.shape<1>, !tla.coord<1> -> !tla.tensor<!tla.layout<!tla.shape<1>, !tla.stride<1>, !tla.shape<1>, RowMajor>, !tla.coord<1>, !tla.ptr<f32, ub, 256>>
    %18 = tla.tensor_ptr_with_coord %17 : !tla.tensor<!tla.layout<!tla.shape<1>, !tla.stride<1>, !tla.shape<1>, RowMajor>, !tla.coord<1>, !tla.ptr<f32, ub, 256>> -> !tla.ptr<f32, ub, 256>
    %19 = tla.make_coord -> !tla.coord<0>
    %20 = tla.make_tensor %18, %3, %19 : !tla.ptr<f32, ub, 256>, !tla.layout<!tla.shape<64>, !tla.stride<1>, !tla.shape<64>, RowMajor>, !tla.coord<0> -> !tla.tensor<!tla.layout<!tla.shape<64>, !tla.stride<1>, !tla.shape<64>, RowMajor>, !tla.coord<0>, !tla.ptr<f32, ub, 256>>
    %21 = tla.make_coord -> !tla.coord<0>
    %22 = tla.make_tensor %10, %3, %21 : !tla.ptr<f32, ub, 256>, !tla.layout<!tla.shape<64>, !tla.stride<1>, !tla.shape<64>, RowMajor>, !tla.coord<0> -> !tla.tensor<!tla.layout<!tla.shape<64>, !tla.stride<1>, !tla.shape<64>, RowMajor>, !tla.coord<0>, !tla.ptr<f32, ub, 256>>
    %23 = tla.make_coord -> !tla.coord<0>
    %24 = tla.make_tensor %11, %3, %23 : !tla.ptr<f32, ub, 256>, !tla.layout<!tla.shape<64>, !tla.stride<1>, !tla.shape<64>, RowMajor>, !tla.coord<0> -> !tla.tensor<!tla.layout<!tla.shape<64>, !tla.stride<1>, !tla.shape<64>, RowMajor>, !tla.coord<0>, !tla.ptr<f32, ub, 256>>
    "tla.vector"() ({
      "tla.vec.func"() ({
        %cst = arith.constant 1.100000e+01 : f32
        %25 = tla.full %cst : f32 -> !tla.vector<64xf32>
        %c0_i32 = arith.constant 0 : i32
        %c1_i32 = arith.constant 1 : i32
        %c1_i32_0 = arith.constant 1 : i32
        %26 = scf.for %arg1 = %c0_i32 to %c1_i32 step %c1_i32_0 iter_args(%arg2 = %20) -> (!tla.tensor<!tla.layout<!tla.shape<64>, !tla.stride<1>, !tla.shape<64>, RowMajor>, !tla.coord<0>, !tla.ptr<f32, ub, 256>>)  : i32 {
          %c64_i32_1 = arith.constant 64 : i32
          %next_ptr_2, %ureg_out_3 = tla.store_unalign_update %arg2, %25 post_update_stride %c64_i32_1 : <!tla.layout<!tla.shape<64>, !tla.stride<1>, !tla.shape<64>, RowMajor>, !tla.coord<0>, !tla.ptr<f32, ub, 256>>, !tla.vector<64xf32> post_update_stride i32 -> !tla.ptr<f32, ub, 256>, !tla.ureg
          %33 = tla.make_shape -> !tla.shape<64>
          %34 = tla.make_stride -> !tla.stride<1>
          %35 = tla.make_shape -> !tla.shape<64>
          %36 = tla.make_layout %33, %34 origin %35 : !tla.shape<64>, !tla.stride<1> origin !tla.shape<64> -> !tla.layout<!tla.shape<64>, !tla.stride<1>, !tla.shape<64>, RowMajor>
          %37 = tla.make_coord -> !tla.coord<0>
          %38 = tla.make_tensor %next_ptr_2, %36, %37 : !tla.ptr<f32, ub, 256>, !tla.layout<!tla.shape<64>, !tla.stride<1>, !tla.shape<64>, RowMajor>, !tla.coord<0> -> !tla.tensor<!tla.layout<!tla.shape<64>, !tla.stride<1>, !tla.shape<64>, RowMajor>, !tla.coord<0>, !tla.ptr<f32, ub, 256>>
          tla.local_mem_bar 1
          %result = tla.load %22 {unaligned_ub_access} : <!tla.layout<!tla.shape<64>, !tla.stride<1>, !tla.shape<64>, RowMajor>, !tla.coord<0>, !tla.ptr<f32, ub, 256>> -> !tla.vector<64xf32>
          tla.store %24, %result : <!tla.layout<!tla.shape<64>, !tla.stride<1>, !tla.shape<64>, RowMajor>, !tla.coord<0>, !tla.ptr<f32, ub, 256>>, !tla.vector<64xf32>
          scf.yield %38 : !tla.tensor<!tla.layout<!tla.shape<64>, !tla.stride<1>, !tla.shape<64>, RowMajor>, !tla.coord<0>, !tla.ptr<f32, ub, 256>>
        }
        %c64_i32 = arith.constant 64 : i32
        %next_ptr, %ureg_out = tla.store_unalign_update %26, %25 post_update_stride %c64_i32 : <!tla.layout<!tla.shape<64>, !tla.stride<1>, !tla.shape<64>, RowMajor>, !tla.coord<0>, !tla.ptr<f32, ub, 256>>, !tla.vector<64xf32> post_update_stride i32 -> !tla.ptr<f32, ub, 256>, !tla.ureg
        %27 = tla.make_shape -> !tla.shape<64>
        %28 = tla.make_stride -> !tla.stride<1>
        %29 = tla.make_shape -> !tla.shape<64>
        %30 = tla.make_layout %27, %28 origin %29 : !tla.shape<64>, !tla.stride<1> origin !tla.shape<64> -> !tla.layout<!tla.shape<64>, !tla.stride<1>, !tla.shape<64>, RowMajor>
        %31 = tla.make_coord -> !tla.coord<0>
        %32 = tla.make_tensor %next_ptr, %30, %31 : !tla.ptr<f32, ub, 256>, !tla.layout<!tla.shape<64>, !tla.stride<1>, !tla.shape<64>, RowMajor>, !tla.coord<0> -> !tla.tensor<!tla.layout<!tla.shape<64>, !tla.stride<1>, !tla.shape<64>, RowMajor>, !tla.coord<0>, !tla.ptr<f32, ub, 256>>
      }) {mode = "simd"} : () -> ()
      tla.copy %24, %arg0 : !tla.tensor<!tla.layout<!tla.shape<64>, !tla.stride<1>, !tla.shape<64>, RowMajor>, !tla.coord<0>, !tla.ptr<f32, ub, 256>>, !tla.tensor<!tla.layout<!tla.shape<64>, !tla.stride<1>, !tla.shape<64>, RowMajor>, !tla.coord<0>, !tla.ptr<f32, gm, 4>>
    }) : () -> ()
    tla.return
  }
}

// The ptr_with_coord read window overlaps the store's allocation: POST must
// run before the local_mem_bar / load, not after the loop.
// CHECK-LABEL: func.func @ptr_with_coord_window
// CHECK:         scf.for
// CHECK:         tla.store_unalign_update
// CHECK:         tla.store_unalign_post
// CHECK-NEXT:    tla.local_mem_bar 1
// CHECK-NEXT:    %[[READ:.*]] = tla.load

// Independent allocations: the store stream may still eliminate the per-trip
// POST and flush once after the loop.
// CHECK-LABEL: func.func @separate_windows
// CHECK:         scf.for
// CHECK:         tla.store_unalign_update
// CHECK-NOT:     tla.store_unalign_post
// CHECK:         scf.yield
// CHECK:         tla.store_unalign_post
