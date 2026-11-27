// RUN: %tla_compile %s -o - | %filecheck %s -D'$DTYPE=uint8_t'
// RUN: sed 's/ui8/i32/g' %s | %tla_compile - -o - | %filecheck %s -D'$DTYPE=int32_t'
// RUN: sed 's/ui8/bf16/g' %s | %tla_compile - -o - | %filecheck %s -D'$DTYPE=bf16'
// RUN: sed 's/ui8/f64/g' %s | %tla_compile - -o - | %filecheck %s -D'$DTYPE=double'
// RUN: sed 's/ui8/f8E4M3FN/g' %s | %tla_compile - -o - | %filecheck %s -D'$DTYPE=fp8_e4m3fn_t'
// RUN: sed 's/ui8/!tla.f4e2m1/g' %s | %tla_compile - -o - | %filecheck %s -D'$DTYPE=float4_e2m1x2_t'
// RUN: sed 's/ui8/!tla.i4x2/g' %s | %tla_compile - -o - | %filecheck %s -D'$DTYPE=int8_t'
// RUN: sed 's/!tla.stride<8,1>/!tla.stride<32,1>/g' %s > %t.padded.mlir
// RUN: not %tla_compile %t.padded.mlir -o - 2>&1 | %filecheck %s --check-prefix=PADDED
// RUN: sed 's/ui8/!tla.f4e2m1/g; s/!tla.stride<96,1>/!tla.stride<95,1>/g' %s > %t.fp4-odd.mlir
// RUN: not %tla_compile %t.fp4-odd.mlir -o - 2>&1 | %filecheck %s --check-prefix=FP4-ODD

!gm = !tla.tensor<!tla.layout<!tla.shape<17,8>, !tla.stride<96,1>, !tla.shape<17,8>, RowMajor>, !tla.coord<0,0>, !tla.ptr<ui8, gm, 4>>
!ub = !tla.tensor<!tla.layout<!tla.shape<17,8>, !tla.stride<8,1>, !tla.shape<17,8>, RowMajor>, !tla.coord<0,0>, !tla.ptr<ui8, ub, 256>>

module {
  tla.func @compact_ub2gm(%dst: !gm) {
    %ptr = tla.alloc_ptr {size_bytes = 2048} -> !tla.ptr<ui8, ub, 256>
    %shape = tla.make_shape -> !tla.shape<17,8>
    %stride = tla.make_stride -> !tla.stride<8,1>
    %layout = tla.make_layout %shape, %stride : !tla.shape<17,8>, !tla.stride<8,1> -> !tla.layout<!tla.shape<17,8>, !tla.stride<8,1>, !tla.shape<17,8>, RowMajor>
    %coord = tla.make_coord -> !tla.coord<0,0>
    %src = tla.make_tensor %ptr, %layout, %coord : !tla.ptr<ui8, ub, 256>, !tla.layout<!tla.shape<17,8>, !tla.stride<8,1>, !tla.shape<17,8>, RowMajor>, !tla.coord<0,0> -> !ub
    "tla.vector"() ({
      tla.copy %dst, %src {compact} : !gm, !ub
    }) : () -> ()
    tla.return
  }
}

// CHECK-LABEL: func.func private @copy_ub_RowMajor_to_gm_RowMajor_
// CHECK-SAME: [[$DTYPE]]_compact(
// CHECK-LABEL: func.func @compact_ub2gm
// CHECK: call @copy_ub_RowMajor_to_gm_RowMajor_[[$DTYPE]]_compact(
// PADDED: compact requires the UB row stride to equal the valid row width
// FP4-ODD: compact FP4 copy requires statically even row width, row strides and column coordinates
