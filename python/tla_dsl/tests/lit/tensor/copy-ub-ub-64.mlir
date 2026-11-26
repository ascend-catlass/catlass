// RUN: %tla_compile %s -o - | %filecheck %s --check-prefixes=CHECK,SIGNED
// RUN: sed 's/xi64/xui64/g;s/<i64,/<ui64,/g' %s | %tla_compile - -o - | %filecheck %s --check-prefixes=CHECK,UNSIGNED
// RUN: sed 's/i64/f64/g' %s | %tla_compile - -o - | %filecheck %s --check-prefixes=CHECK,FLOAT

// UB→UB RowMajor b64 (i64/u64/f64) is a supported MOV.UB.TO.UB route.
// SIGNED: copy_ub_RowMajor_to_ub_RowMajor_int64_t
// UNSIGNED: copy_ub_RowMajor_to_ub_RowMajor_uint64_t
// FLOAT: copy_ub_RowMajor_to_ub_RowMajor_double
// CHECK-SAME: hivm.func_core_type = #hivm.func_core_type<AIV>
// SIGNED: call @copy_ub_RowMajor_to_ub_RowMajor_int64_t
// UNSIGNED: call @copy_ub_RowMajor_to_ub_RowMajor_uint64_t
// FLOAT: call @copy_ub_RowMajor_to_ub_RowMajor_double

!tile = !tla.tensor<!tla.layout<!tla.shape<32>, !tla.stride<1>, !tla.shape<32>, RowMajor>, !tla.coord<0>, !tla.ptr<i64, ub, 256>>
!layout = !tla.layout<!tla.shape<32>, !tla.stride<1>, !tla.shape<32>, RowMajor>
!ptr = !tla.ptr<i64, ub, 256>

module {
  tla.func @copy_ub_ub_b64() {
    %src = tla.alloc_ptr {size_bytes = 256} -> !ptr
    %dst = tla.alloc_ptr {size_bytes = 256} -> !ptr
    %shape = tla.make_shape -> !tla.shape<32>
    %stride = tla.make_stride -> !tla.stride<1>
    %layout = tla.make_layout %shape, %stride : !tla.shape<32>, !tla.stride<1> -> !layout
    %coord = tla.make_coord -> !tla.coord<0>
    %a = tla.make_tensor %src, %layout, %coord : !ptr, !layout, !tla.coord<0> -> !tile
    %b = tla.make_tensor %dst, %layout, %coord : !ptr, !layout, !tla.coord<0> -> !tile
    "tla.vector"() ({
      tla.copy %b, %a : !tile, !tile
    }) : () -> ()
    tla.return
  }
}
