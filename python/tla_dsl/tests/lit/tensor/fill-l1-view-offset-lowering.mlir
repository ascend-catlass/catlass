// RUN: %tla_compile %s -o %t
// RUN: %filecheck %s --check-prefix=CHECK < %t
// RUN: %filecheck %s --check-prefix=F16 < %t
// RUN: %filecheck %s --check-prefix=F32 < %t
//
// tla.fill on a tile_view of an L1 zN tile (kernel: 32x64 zN tile, view
// (16,32) at tile coord (1,1) = element offset (16,32), fill requested at
// local coord (0,8) with extent (16,8)).
//
// The fill coord that reaches the runtime call is the EFFECTIVE coord
// (caller local coord + the view's own coord = (16,40)), because the bc
// wrapper places the fill relative to the buffer base without adding the
// descriptor coord. The extent is cropped by the front end against the
// view-local origin (16,32) and passes through unchanged. The view's own
// coord (16,32) and origin (16,32) travel in the 12-field descriptor.

module {
  tla.func @_kernel() {
    %0 = tla.make_shape -> !tla.shape<(16,2),(32,2)>
    %1 = tla.make_stride -> !tla.stride<(32,1024),(1,1024)>
    %2 = tla.make_shape -> !tla.shape<32,64>
    %3 = tla.make_layout %0, %1 origin %2 {layoutTag = #tla.layout_tag<zN>} : !tla.shape<(16,2),(32,2)>, !tla.stride<(32,1024),(1,1024)> origin !tla.shape<32,64> -> !tla.layout<!tla.shape<(16,2),(32,2)>, !tla.stride<(32,1024),(1,1024)>, !tla.shape<32,64>, zN>
    %4 = tla.alloc_ptr{size_bytes = 4096} -> !tla.ptr<f8E4M3FN, l1, 512>
    %5 = tla.make_coord -> !tla.coord<0,0>
    %6 = tla.make_tensor %4, %3, %5 : !tla.ptr<f8E4M3FN, l1, 512>, !tla.layout<!tla.shape<(16,2),(32,2)>, !tla.stride<(32,1024),(1,1024)>, !tla.shape<32,64>, zN>, !tla.coord<0,0> -> !tla.tensor<!tla.layout<!tla.shape<(16,2),(32,2)>, !tla.stride<(32,1024),(1,1024)>, !tla.shape<32,64>, zN>, !tla.coord<0,0>, !tla.ptr<f8E4M3FN, l1, 512>>
    %7 = tla.make_shape -> !tla.shape<16,32>
    %8 = tla.make_coord -> !tla.coord<1,1>
    %9 = tla.make_coord -> !tla.coord<16,32>
    %10 = tla.tile_view %6, %7, %9 : !tla.tensor<!tla.layout<!tla.shape<(16,2),(32,2)>, !tla.stride<(32,1024),(1,1024)>, !tla.shape<32,64>, zN>, !tla.coord<0,0>, !tla.ptr<f8E4M3FN, l1, 512>>, !tla.shape<16,32>, !tla.coord<16,32> -> !tla.tensor<!tla.layout<!tla.shape<(16,1),(32,1)>, !tla.stride<(32,1024),(1,1024)>, !tla.shape<16,32>, zN>, !tla.coord<16,32>, !tla.ptr<f8E4M3FN, l1, 512>>
    tla.cube {
      %11 = tla.make_coord -> !tla.coord<0,8>
      %12 = tla.make_shape -> !tla.shape<16,8>
      %13 = tla.make_coord -> !tla.coord<16,40>
      %c0_i32 = arith.constant 0 : i32
      %14 = tla.make_coord -> !tla.coord<16,40>
      %15 = tla.make_shape -> !tla.shape<16,8>
      tla.fill %10, %c0_i32, %14, %15 : !tla.tensor<!tla.layout<!tla.shape<(16,1),(32,1)>, !tla.stride<(32,1024),(1,1024)>, !tla.shape<16,32>, zN>, !tla.coord<16,32>, !tla.ptr<f8E4M3FN, l1, 512>>, i32, <16,40>, <16,8>
    }
    tla.return
  }

  tla.func @_kernel_f16() {
    %0 = tla.make_shape -> !tla.shape<(16,2),(16,4)>
    %1 = tla.make_stride -> !tla.stride<(32,1024),(1,1024)>
    %2 = tla.make_shape -> !tla.shape<32,64>
    %3 = tla.make_layout %0, %1 origin %2 {layoutTag = #tla.layout_tag<zN>} : !tla.shape<(16,2),(16,4)>, !tla.stride<(32,1024),(1,1024)> origin !tla.shape<32,64> -> !tla.layout<!tla.shape<(16,2),(16,4)>, !tla.stride<(32,1024),(1,1024)>, !tla.shape<32,64>, zN>
    %4 = tla.alloc_ptr{size_bytes = 4096} -> !tla.ptr<f16, l1, 512>
    %5 = tla.make_coord -> !tla.coord<0,0>
    %6 = tla.make_tensor %4, %3, %5 : !tla.ptr<f16, l1, 512>, !tla.layout<!tla.shape<(16,2),(16,4)>, !tla.stride<(32,1024),(1,1024)>, !tla.shape<32,64>, zN>, !tla.coord<0,0> -> !tla.tensor<!tla.layout<!tla.shape<(16,2),(16,4)>, !tla.stride<(32,1024),(1,1024)>, !tla.shape<32,64>, zN>, !tla.coord<0,0>, !tla.ptr<f16, l1, 512>>
    %7 = tla.make_shape -> !tla.shape<16,32>
    %8 = tla.make_coord -> !tla.coord<1,1>
    %9 = tla.make_coord -> !tla.coord<16,32>
    %10 = tla.tile_view %6, %7, %9 : !tla.tensor<!tla.layout<!tla.shape<(16,2),(16,4)>, !tla.stride<(32,1024),(1,1024)>, !tla.shape<32,64>, zN>, !tla.coord<0,0>, !tla.ptr<f16, l1, 512>>, !tla.shape<16,32>, !tla.coord<16,32> -> !tla.tensor<!tla.layout<!tla.shape<(16,1),(32,1)>, !tla.stride<(32,1024),(1,1024)>, !tla.shape<16,32>, zN>, !tla.coord<16,32>, !tla.ptr<f16, l1, 512>>
    tla.cube {
      %11 = tla.make_coord -> !tla.coord<0,8>
      %12 = tla.make_shape -> !tla.shape<16,8>
      %13 = tla.make_coord -> !tla.coord<16,40>
      %c0_i32 = arith.constant 0 : i32
      %14 = tla.make_coord -> !tla.coord<16,40>
      %15 = tla.make_shape -> !tla.shape<16,8>
      tla.fill %10, %c0_i32, %14, %15 : !tla.tensor<!tla.layout<!tla.shape<(16,1),(32,1)>, !tla.stride<(32,1024),(1,1024)>, !tla.shape<16,32>, zN>, !tla.coord<16,32>, !tla.ptr<f16, l1, 512>>, i32, <16,40>, <16,8>
    }
    tla.return
  }

  tla.func @_kernel_f32() {
    %0 = tla.make_shape -> !tla.shape<(16,2),(8,8)>
    %1 = tla.make_stride -> !tla.stride<(32,1024),(1,1024)>
    %2 = tla.make_shape -> !tla.shape<32,64>
    %3 = tla.make_layout %0, %1 origin %2 {layoutTag = #tla.layout_tag<zN>} : !tla.shape<(16,2),(8,8)>, !tla.stride<(32,1024),(1,1024)> origin !tla.shape<32,64> -> !tla.layout<!tla.shape<(16,2),(8,8)>, !tla.stride<(32,1024),(1,1024)>, !tla.shape<32,64>, zN>
    %4 = tla.alloc_ptr{size_bytes = 4096} -> !tla.ptr<f32, l1, 512>
    %5 = tla.make_coord -> !tla.coord<0,0>
    %6 = tla.make_tensor %4, %3, %5 : !tla.ptr<f32, l1, 512>, !tla.layout<!tla.shape<(16,2),(8,8)>, !tla.stride<(32,1024),(1,1024)>, !tla.shape<32,64>, zN>, !tla.coord<0,0> -> !tla.tensor<!tla.layout<!tla.shape<(16,2),(8,8)>, !tla.stride<(32,1024),(1,1024)>, !tla.shape<32,64>, zN>, !tla.coord<0,0>, !tla.ptr<f32, l1, 512>>
    %7 = tla.make_shape -> !tla.shape<16,32>
    %8 = tla.make_coord -> !tla.coord<1,1>
    %9 = tla.make_coord -> !tla.coord<16,32>
    %10 = tla.tile_view %6, %7, %9 : !tla.tensor<!tla.layout<!tla.shape<(16,2),(8,8)>, !tla.stride<(32,1024),(1,1024)>, !tla.shape<32,64>, zN>, !tla.coord<0,0>, !tla.ptr<f32, l1, 512>>, !tla.shape<16,32>, !tla.coord<16,32> -> !tla.tensor<!tla.layout<!tla.shape<(16,1),(32,1)>, !tla.stride<(32,1024),(1,1024)>, !tla.shape<16,32>, zN>, !tla.coord<16,32>, !tla.ptr<f32, l1, 512>>
    tla.cube {
      %11 = tla.make_coord -> !tla.coord<0,8>
      %12 = tla.make_shape -> !tla.shape<16,8>
      %13 = tla.make_coord -> !tla.coord<16,40>
      %c0_i32 = arith.constant 0 : i32
      %14 = tla.make_coord -> !tla.coord<16,40>
      %15 = tla.make_shape -> !tla.shape<16,8>
      tla.fill %10, %c0_i32, %14, %15 : !tla.tensor<!tla.layout<!tla.shape<(16,1),(32,1)>, !tla.stride<(32,1024),(1,1024)>, !tla.shape<16,32>, zN>, !tla.coord<16,32>, !tla.ptr<f32, l1, 512>>, i32, <16,40>, <16,8>
    }
    tla.return
  }
}

// The runtime template: f8E4M3FN maps to the fp8_e4m3fn_t suffix, L1 lands on the AIC
// core, and the template is inlinable with a C interface.
// CHECK: func.func private @fill_l1_zN_fp8_e4m3fn_t(memref<?xf8E4M3FN, strided<[?], offset: ?>, #hivm.address_space<cbuf>>, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i32)
// CHECK-SAME: hacc.always_inline
// CHECK-SAME: hivm.func_core_type = #hivm.func_core_type<AIC>
// CHECK-SAME: llvm.emit_c_interface
// CHECK-LABEL: func.func @_kernel()
// Fill region tail of the payload: crd0=16, crd1=40 (own 32 + local 8),
// ext0=16, ext1=8, then the i32 bit pattern.
// CHECK-DAG: %[[T8:.+]] = llvm.mlir.constant(8 : i64) : i64
// CHECK-DAG: %[[T40:.+]] = llvm.mlir.constant(40 : i64) : i64
// CHECK-DAG: %[[T16:.+]] = llvm.mlir.constant(16 : i64) : i64
// CHECK-DAG: %[[VAL0:.+]] = llvm.mlir.constant(0 : i32) : i32
// CHECK: call @fill_l1_zN_fp8_e4m3fn_t
// CHECK-SAME: %[[T16]], %[[T40]], %[[T16]], %[[T8]], %[[VAL0]])
// CHECK-NOT: tla.fill

// F16: func.func private @fill_l1_zN_half(memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<cbuf>>, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i32)
// F16-SAME: hacc.always_inline
// F16-SAME: hivm.func_core_type = #hivm.func_core_type<AIC>
// F16-SAME: llvm.emit_c_interface
// F16-LABEL: func.func @_kernel_f16()
// F16-DAG: %[[T8:.+]] = llvm.mlir.constant(8 : i64) : i64
// F16-DAG: %[[T40:.+]] = llvm.mlir.constant(40 : i64) : i64
// F16-DAG: %[[T16:.+]] = llvm.mlir.constant(16 : i64) : i64
// F16-DAG: %[[VAL0:.+]] = llvm.mlir.constant(0 : i32) : i32
// F16: call @fill_l1_zN_half
// F16-SAME: %[[T16]], %[[T40]], %[[T16]], %[[T8]], %[[VAL0]])
// F16-NOT: tla.fill

// F32: func.func private @fill_l1_zN_float(memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<cbuf>>, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i32)
// F32-SAME: hacc.always_inline
// F32-SAME: hivm.func_core_type = #hivm.func_core_type<AIC>
// F32-SAME: llvm.emit_c_interface
// F32-LABEL: func.func @_kernel_f32()
// F32-DAG: %[[T8:.+]] = llvm.mlir.constant(8 : i64) : i64
// F32-DAG: %[[T40:.+]] = llvm.mlir.constant(40 : i64) : i64
// F32-DAG: %[[T16:.+]] = llvm.mlir.constant(16 : i64) : i64
// F32-DAG: %[[VAL0:.+]] = llvm.mlir.constant(0 : i32) : i32
// F32: call @fill_l1_zN_float
// F32-SAME: %[[T16]], %[[T40]], %[[T16]], %[[T8]], %[[VAL0]])
// F32-NOT: tla.fill
