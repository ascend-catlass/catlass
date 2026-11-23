// RUN: %tla_compile %s -o - | %filecheck %s --check-prefixes=CHECK,SIGNED
// RUN: sed 's/ptr<i64/ptr<ui64/g' %s | %tla_compile - -o - | %filecheck %s --check-prefixes=CHECK,UNSIGNED
// RUN: sed 's/ptr<i64/ptr<f64/g' %s | %tla_compile - -o - | %filecheck %s --check-prefixes=CHECK,FLOAT
// RUN: sed 's/ptr<i64/ptr<f32/g' %s | %tla_compile - -o - | %filecheck %s --check-prefixes=CHECK,F32
// RUN: sed 's/ptr<i64/ptr<i1/g' %s | %tla_compile - -o - | %filecheck %s --check-prefixes=CHECK,BOOL
// RUN: sed 's/tla.copy %ub_dst, %gm_src : !ub, !gm/tla.copy %gm_src, %ub_dst : !gm, !ub/' %s | not %tla_compile - -o - 2>&1 | %filecheck %s --check-prefix=ROUTE-ERR

!layout = !tla.layout<!tla.shape<32>, !tla.stride<1>, !tla.shape<32>, VectorLayout>
!gm = !tla.tensor<!layout, !tla.coord<0>, !tla.ptr<i64, gm, 8>>
!ub = !tla.tensor<!layout, !tla.coord<0>, !tla.ptr<i64, ub, 256>>

module {
  tla.func @copy_vector_layout(%gm_src: !gm) {
    %ptr = tla.alloc_ptr{size_bytes = 256} -> !tla.ptr<i64, ub, 256>
    %shape = tla.make_shape -> !tla.shape<32>
    %stride = tla.make_stride -> !tla.stride<1>
    %layout = tla.make_layout %shape, %stride : !tla.shape<32>, !tla.stride<1> -> !layout
    %coord = tla.make_coord -> !tla.coord<0>
    %ub_dst = tla.make_tensor %ptr, %layout, %coord : !tla.ptr<i64, ub, 256>, !layout, !tla.coord<0> -> !ub
    "tla.vector"() ({
      tla.copy %ub_dst, %gm_src : !ub, !gm
    }) : () -> ()
    tla.return
  }
}

// SIGNED: func.func private @copy_gm_VectorLayout_to_ub_VectorLayout_int64_t
// UNSIGNED: func.func private @copy_gm_VectorLayout_to_ub_VectorLayout_uint64_t
// FLOAT: func.func private @copy_gm_VectorLayout_to_ub_VectorLayout_double
// F32: func.func private @copy_gm_VectorLayout_to_ub_VectorLayout_float
// BOOL: func.func private @copy_gm_VectorLayout_to_ub_VectorLayout_bool
// CHECK-SAME: hivm.func_core_type = #hivm.func_core_type<AIV>
// SIGNED: call @copy_gm_VectorLayout_to_ub_VectorLayout_int64_t
// UNSIGNED: call @copy_gm_VectorLayout_to_ub_VectorLayout_uint64_t
// FLOAT: call @copy_gm_VectorLayout_to_ub_VectorLayout_double
// F32: call @copy_gm_VectorLayout_to_ub_VectorLayout_float
// BOOL: call @copy_gm_VectorLayout_to_ub_VectorLayout_bool
// ROUTE-ERR: failed to legalize operation 'tla.copy'
