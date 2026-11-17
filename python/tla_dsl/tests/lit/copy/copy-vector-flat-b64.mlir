// Flat VectorLayout GM->L1 for 64-bit element types. Callee suffixes stay out of
// the shared copyRuntimeElemSuffix map and are resolved only on this route:
// ui64 -> uint64_t, i64 -> int64_t, f64 -> double.

// RUN: split-file %s %t
// RUN: %tla_compile %t/u64.mlir -o - 2>&1 | %filecheck %s --check-prefix=U64
// RUN: %tla_compile %t/i64.mlir -o - 2>&1 | %filecheck %s --check-prefix=I64
// RUN: %tla_compile %t/f64.mlir -o - 2>&1 | %filecheck %s --check-prefix=F64

//--- u64.mlir

!gm = !tla.tensor<!tla.layout<!tla.shape<32>, !tla.stride<1>, !tla.shape<32>, VectorLayout>, !tla.coord<0>, !tla.ptr<ui64, gm, 8>>
!l1 = !tla.tensor<!tla.layout<!tla.shape<32>, !tla.stride<1>, !tla.shape<32>, VectorLayout>, !tla.coord<0>, !tla.ptr<ui64, l1, 512>>

module {
  "tla.func"() ({
  ^bb0(%arg0: !gm):
    %0 = "tla.alloc_ptr"() {size_bytes = 256 : i64} : () -> !tla.ptr<ui64, l1, 512>
    %1 = "tla.make_tensor_like"(%0, %arg0) {layoutTag = #tla.layout_tag<VectorLayout>} : (!tla.ptr<ui64, l1, 512>, !gm) -> !l1
    "tla.cube"() ({
      "tla.copy"(%1, %arg0) : (!l1, !gm) -> ()
    }) : () -> ()
    "tla.return"() : () -> ()
  }) {function_type = (!gm) -> (), sym_name = "vector_flat_u64"} : () -> ()
}

// U64-DAG: func.func private @copy_gm_VectorLayout_to_l1_VectorLayout_uint64_t
// U64: call @copy_gm_VectorLayout_to_l1_VectorLayout_uint64_t

//--- i64.mlir

!gm = !tla.tensor<!tla.layout<!tla.shape<32>, !tla.stride<1>, !tla.shape<32>, VectorLayout>, !tla.coord<0>, !tla.ptr<i64, gm, 8>>
!l1 = !tla.tensor<!tla.layout<!tla.shape<32>, !tla.stride<1>, !tla.shape<32>, VectorLayout>, !tla.coord<0>, !tla.ptr<i64, l1, 512>>

module {
  "tla.func"() ({
  ^bb0(%arg0: !gm):
    %0 = "tla.alloc_ptr"() {size_bytes = 256 : i64} : () -> !tla.ptr<i64, l1, 512>
    %1 = "tla.make_tensor_like"(%0, %arg0) {layoutTag = #tla.layout_tag<VectorLayout>} : (!tla.ptr<i64, l1, 512>, !gm) -> !l1
    "tla.cube"() ({
      "tla.copy"(%1, %arg0) : (!l1, !gm) -> ()
    }) : () -> ()
    "tla.return"() : () -> ()
  }) {function_type = (!gm) -> (), sym_name = "vector_flat_i64"} : () -> ()
}

// I64-DAG: func.func private @copy_gm_VectorLayout_to_l1_VectorLayout_int64_t
// I64: call @copy_gm_VectorLayout_to_l1_VectorLayout_int64_t

//--- f64.mlir

!gm = !tla.tensor<!tla.layout<!tla.shape<32>, !tla.stride<1>, !tla.shape<32>, VectorLayout>, !tla.coord<0>, !tla.ptr<f64, gm, 8>>
!l1 = !tla.tensor<!tla.layout<!tla.shape<32>, !tla.stride<1>, !tla.shape<32>, VectorLayout>, !tla.coord<0>, !tla.ptr<f64, l1, 512>>

module {
  "tla.func"() ({
  ^bb0(%arg0: !gm):
    %0 = "tla.alloc_ptr"() {size_bytes = 256 : i64} : () -> !tla.ptr<f64, l1, 512>
    %1 = "tla.make_tensor_like"(%0, %arg0) {layoutTag = #tla.layout_tag<VectorLayout>} : (!tla.ptr<f64, l1, 512>, !gm) -> !l1
    "tla.cube"() ({
      "tla.copy"(%1, %arg0) : (!l1, !gm) -> ()
    }) : () -> ()
    "tla.return"() : () -> ()
  }) {function_type = (!gm) -> (), sym_name = "vector_flat_f64"} : () -> ()
}

// F64-DAG: func.func private @copy_gm_VectorLayout_to_l1_VectorLayout_double
// F64: call @copy_gm_VectorLayout_to_l1_VectorLayout_double
