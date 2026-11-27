// RUN: %tla_compile %s -o - | %filecheck %s -D'$DTYPE=uint8_t'
// RUN: sed 's/ui8/i32/g' %s | %tla_compile - -o - | %filecheck %s -D'$DTYPE=int32_t'
// RUN: sed 's/ui8/!tla.f4e2m1/g' %s | %tla_compile - -o - | %filecheck %s -D'$DTYPE=float4_e2m1x2_t'
// RUN: sed 's/!tla.stride<6,1>/!tla.stride<32,1>/g' %s | not %tla_compile - -o - 2>&1 | %filecheck %s --check-prefix=PADDED
// RUN: sed 's/!tla.stride<12,1>/!tla.stride<5,1>/g' %s | not %tla_compile - -o - 2>&1 | %filecheck %s --check-prefix=SMALL-GM-STRIDE
// RUN: sed 's/ui8/!tla.f4e2m1/g; s/!tla.stride<12,1>/!tla.stride<11,1>/g' %s | not %tla_compile - -o - 2>&1 | %filecheck %s --check-prefix=FP4-ODD
// RUN: sed '/^[[:space:]]*tla\.copy /s/ : / {compact} : /' %S/copy-gm-ub-pad-params.mlir | not %tla_compile - -o - 2>&1 | %filecheck %s --check-prefix=PAD-PARAMS

!gm = !tla.tensor<!tla.layout<!tla.shape<3,6>, !tla.stride<12,1>, !tla.shape<3,6>, RowMajor>, !tla.coord<0,0>, !tla.ptr<ui8, gm, 4>>
!layout = !tla.layout<!tla.shape<3,6>, !tla.stride<6,1>, !tla.shape<3,6>, RowMajor>
!ub = !tla.tensor<!layout, !tla.coord<0,0>, !tla.ptr<ui8, ub, 256>>

module {
  tla.func @compact_gm2ub(%src: !gm) {
    %ptr = tla.alloc_ptr {size_bytes = 256} -> !tla.ptr<ui8, ub, 256>
    %shape = tla.make_shape -> !tla.shape<3,6>
    %stride = tla.make_stride -> !tla.stride<6,1>
    %layout = tla.make_layout %shape, %stride : !tla.shape<3,6>, !tla.stride<6,1> -> !layout
    %coord = tla.make_coord -> !tla.coord<0,0>
    %dst = tla.make_tensor %ptr, %layout, %coord : !tla.ptr<ui8, ub, 256>, !layout, !tla.coord<0,0> -> !ub
    "tla.vector"() ({
      tla.copy %dst, %src {compact} : !ub, !gm
    }) : () -> ()
    tla.return
  }
}

// CHECK-LABEL: func.func private @copy_gm_RowMajor_to_ub_RowMajor_
// CHECK-SAME: [[$DTYPE]]_compact({{.*}}#hivm.address_space<ub>>{{(, i64){25}\)}}
// CHECK-LABEL: func.func @compact_gm2ub
// CHECK: call @copy_gm_RowMajor_to_ub_RowMajor_[[$DTYPE]]_compact(
// PADDED: compact requires the UB row stride to equal the valid row width
// SMALL-GM-STRIDE: must have non-overlapping rows
// FP4-ODD: compact FP4 copy requires statically even row width, row strides and column coordinates
// PAD-PARAMS: compact copy does not accept a padding parameter operand
