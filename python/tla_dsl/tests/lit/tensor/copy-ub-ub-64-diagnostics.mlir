// RUN: not %tla_compile %s -o - 2>&1 | %filecheck %s
// RUN: sed 's/xi64/xui64/g;s/<i64,/<ui64,/g' %s | not %tla_compile - -o - 2>&1 | %filecheck %s
// RUN: sed 's/i64/f64/g' %s | not %tla_compile - -o - 2>&1 | %filecheck %s

!tile = !tla.tensor<!tla.layout<!tla.shape<32>, !tla.stride<1>, !tla.shape<32>, RowMajor>, !tla.coord<0>, !tla.ptr<i64, ub, 256>>
!layout = !tla.layout<!tla.shape<32>, !tla.stride<1>, !tla.shape<32>, RowMajor>
!ptr = !tla.ptr<i64, ub, 256>

module {
  tla.func @unsupported_copy() {
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

// CHECK: failed to legalize operation 'tla.copy'
