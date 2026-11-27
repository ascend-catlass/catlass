// RUN: %tla_compile %s -o - | %filecheck %s
// Copies with an explicit atomic_mode configure and disable it per transfer.
// The final copy leaves the explicitly patched atomic format active.

module {
  tla.func @atomic_add_ub2gm(%arg0: !tla.tensor<!tla.layout<!tla.shape<64,64>, !tla.stride<64,1>, !tla.shape<64,64>, RowMajor>, !tla.coord<0,0>, !tla.ptr<i32, gm, 4>>) {
    %0 = tla.make_shape -> !tla.shape<16,16>
    %1 = tla.make_coord -> !tla.coord<0,0>
    %2 = tla.make_coord -> !tla.coord<0,0>
    %3 = tla.tile_view %arg0, %0, %2 : !tla.tensor<!tla.layout<!tla.shape<64,64>, !tla.stride<64,1>, !tla.shape<64,64>, RowMajor>, !tla.coord<0,0>, !tla.ptr<i32, gm, 4>>, !tla.shape<16,16>, !tla.coord<0,0> -> !tla.tensor<!tla.layout<!tla.shape<16,16>, !tla.stride<64,1>, !tla.shape<16,16>, RowMajor>, !tla.coord<0,0>, !tla.ptr<i32, gm, 4>>
    %4 = tla.alloc_ptr{size_bytes = 1024} -> !tla.ptr<i8, ub, 256>
    %5 = tla.recast_ptr %4 : !tla.ptr<i8, ub, 256> -> !tla.ptr<i32, ub, 256>
    %6 = tla.make_shape -> !tla.shape<16,16>
    %7 = tla.make_stride -> !tla.stride<16,1>
    %8 = tla.make_layout %6, %7 : !tla.shape<16,16>, !tla.stride<16,1> -> !tla.layout<!tla.shape<16,16>, !tla.stride<16,1>, !tla.shape<16,16>, RowMajor>
    %9 = tla.make_coord -> !tla.coord<0,0>
    %10 = tla.make_tensor %5, %8, %9 : !tla.ptr<i32, ub, 256>, !tla.layout<!tla.shape<16,16>, !tla.stride<16,1>, !tla.shape<16,16>, RowMajor>, !tla.coord<0,0> -> !tla.tensor<!tla.layout<!tla.shape<16,16>, !tla.stride<16,1>, !tla.shape<16,16>, RowMajor>, !tla.coord<0,0>, !tla.ptr<i32, ub, 256>>
    "tla.vector"() ({
      tla.copy %3, %10 {atomic_mode = #tla.atomic_mode<add>} : !tla.tensor<!tla.layout<!tla.shape<16,16>, !tla.stride<64,1>, !tla.shape<16,16>, RowMajor>, !tla.coord<0,0>, !tla.ptr<i32, gm, 4>>, !tla.tensor<!tla.layout<!tla.shape<16,16>, !tla.stride<16,1>, !tla.shape<16,16>, RowMajor>, !tla.coord<0,0>, !tla.ptr<i32, ub, 256>>
      tla.copy %3, %10 {atomic_mode = #tla.atomic_mode<max>} : !tla.tensor<!tla.layout<!tla.shape<16,16>, !tla.stride<64,1>, !tla.shape<16,16>, RowMajor>, !tla.coord<0,0>, !tla.ptr<i32, gm, 4>>, !tla.tensor<!tla.layout<!tla.shape<16,16>, !tla.stride<16,1>, !tla.shape<16,16>, RowMajor>, !tla.coord<0,0>, !tla.ptr<i32, ub, 256>>
      tla.copy %3, %10 {atomic_mode = #tla.atomic_mode<min>} : !tla.tensor<!tla.layout<!tla.shape<16,16>, !tla.stride<64,1>, !tla.shape<16,16>, RowMajor>, !tla.coord<0,0>, !tla.ptr<i32, gm, 4>>, !tla.tensor<!tla.layout<!tla.shape<16,16>, !tla.stride<16,1>, !tla.shape<16,16>, RowMajor>, !tla.coord<0,0>, !tla.ptr<i32, ub, 256>>
      %atomic_format = arith.constant 256 : i64
      tla.arch.set_ctrl %atomic_format {mask = 448 : i64}
      tla.copy %3, %10 : !tla.tensor<!tla.layout<!tla.shape<16,16>, !tla.stride<64,1>, !tla.shape<16,16>, RowMajor>, !tla.coord<0,0>, !tla.ptr<i32, gm, 4>>, !tla.tensor<!tla.layout<!tla.shape<16,16>, !tla.stride<16,1>, !tla.shape<16,16>, RowMajor>, !tla.coord<0,0>, !tla.ptr<i32, ub, 256>>
      %disabled = arith.constant 0 : i64
      tla.arch.set_ctrl %disabled {mask = 448 : i64}
    }) : () -> ()
    tla.return
  }
}

// CHECK-LABEL: func.func private @copy_ub_RowMajor_to_gm_RowMajor_int32_t
// CHECK-SAME: hivm.func_core_type = #hivm.func_core_type<AIV>
// CHECK-LABEL: func.func @atomic_add_ub2gm
// CHECK: hivm.hir.set_ctrl false at ctrl[6]
// CHECK-NEXT: hivm.hir.set_ctrl false at ctrl[7]
// CHECK-NEXT: hivm.hir.set_ctrl true at ctrl[8]
// CHECK-NEXT: hivm.hir.set_ctrl false at ctrl[9]
// CHECK-NEXT: hivm.hir.set_ctrl false at ctrl[10]
// CHECK: call @copy_ub_RowMajor_to_gm_RowMajor_int32_t
// CHECK: hivm.hir.set_ctrl false at ctrl[6]
// CHECK-NEXT: hivm.hir.set_ctrl false at ctrl[7]
// CHECK-NEXT: hivm.hir.set_ctrl false at ctrl[8]
// CHECK: hivm.hir.set_ctrl false at ctrl[6]
// CHECK-NEXT: hivm.hir.set_ctrl false at ctrl[7]
// CHECK-NEXT: hivm.hir.set_ctrl true at ctrl[8]
// CHECK-NEXT: hivm.hir.set_ctrl true at ctrl[9]
// CHECK-NEXT: hivm.hir.set_ctrl false at ctrl[10]
// CHECK: call @copy_ub_RowMajor_to_gm_RowMajor_int32_t
// CHECK: hivm.hir.set_ctrl false at ctrl[6]
// CHECK-NEXT: hivm.hir.set_ctrl false at ctrl[7]
// CHECK-NEXT: hivm.hir.set_ctrl false at ctrl[8]
// CHECK: hivm.hir.set_ctrl false at ctrl[6]
// CHECK-NEXT: hivm.hir.set_ctrl false at ctrl[7]
// CHECK-NEXT: hivm.hir.set_ctrl true at ctrl[8]
// CHECK-NEXT: hivm.hir.set_ctrl false at ctrl[9]
// CHECK-NEXT: hivm.hir.set_ctrl true at ctrl[10]
// CHECK: call @copy_ub_RowMajor_to_gm_RowMajor_int32_t
// CHECK: hivm.hir.set_ctrl false at ctrl[6]
// CHECK-NEXT: hivm.hir.set_ctrl false at ctrl[7]
// CHECK-NEXT: hivm.hir.set_ctrl false at ctrl[8]
// CHECK: SET.CTRL
// CHECK-NOT: hivm.hir.set_ctrl
// CHECK: call @copy_ub_RowMajor_to_gm_RowMajor_int32_t
// CHECK-NOT: hivm.hir.set_ctrl {{.*}} at ctrl[{{[678]}}]
// CHECK: SET.CTRL
// CHECK-NOT: tla.copy
// CHECK-NOT: tla.make_tensor
