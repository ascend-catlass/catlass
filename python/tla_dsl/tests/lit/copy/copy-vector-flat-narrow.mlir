// Flat VectorLayout GM->L1 for narrow continuous element types (i8 / f16 / bf16).
// These resolve through copyRuntimeElemSuffix (unlike b64 special-cases).

// RUN: split-file %s %t
// RUN: %tla_compile %t/i8.mlir -o - 2>&1 | %filecheck %s --check-prefix=I8
// RUN: %tla_compile %t/f16.mlir -o - 2>&1 | %filecheck %s --check-prefix=F16
// RUN: %tla_compile %t/bf16.mlir -o - 2>&1 | %filecheck %s --check-prefix=BF16

//--- i8.mlir

!gm = !tla.tensor<!tla.layout<!tla.shape<256>, !tla.stride<1>, !tla.shape<256>, VectorLayout>, !tla.coord<0>, !tla.ptr<i8, gm, 1>>
!l1 = !tla.tensor<!tla.layout<!tla.shape<256>, !tla.stride<1>, !tla.shape<256>, VectorLayout>, !tla.coord<0>, !tla.ptr<i8, l1, 512>>

module {
  "tla.func"() ({
  ^bb0(%arg0: !gm):
    %0 = "tla.alloc_ptr"() {size_bytes = 256 : i64} : () -> !tla.ptr<i8, l1, 512>
    %1 = "tla.make_tensor_like"(%0, %arg0) {layoutTag = #tla.layout_tag<VectorLayout>} : (!tla.ptr<i8, l1, 512>, !gm) -> !l1
    "tla.cube"() ({
      "tla.copy"(%1, %arg0) : (!l1, !gm) -> ()
    }) : () -> ()
    "tla.return"() : () -> ()
  }) {function_type = (!gm) -> (), sym_name = "vector_flat_i8"} : () -> ()
}

// I8-DAG: func.func private @copy_gm_VectorLayout_to_l1_VectorLayout_int8_t
// I8: call @copy_gm_VectorLayout_to_l1_VectorLayout_int8_t

//--- f16.mlir

!gm = !tla.tensor<!tla.layout<!tla.shape<128>, !tla.stride<1>, !tla.shape<128>, VectorLayout>, !tla.coord<0>, !tla.ptr<f16, gm, 2>>
!l1 = !tla.tensor<!tla.layout<!tla.shape<128>, !tla.stride<1>, !tla.shape<128>, VectorLayout>, !tla.coord<0>, !tla.ptr<f16, l1, 512>>

module {
  "tla.func"() ({
  ^bb0(%arg0: !gm):
    %0 = "tla.alloc_ptr"() {size_bytes = 256 : i64} : () -> !tla.ptr<f16, l1, 512>
    %1 = "tla.make_tensor_like"(%0, %arg0) {layoutTag = #tla.layout_tag<VectorLayout>} : (!tla.ptr<f16, l1, 512>, !gm) -> !l1
    "tla.cube"() ({
      "tla.copy"(%1, %arg0) : (!l1, !gm) -> ()
    }) : () -> ()
    "tla.return"() : () -> ()
  }) {function_type = (!gm) -> (), sym_name = "vector_flat_f16"} : () -> ()
}

// F16-DAG: func.func private @copy_gm_VectorLayout_to_l1_VectorLayout_half
// F16: call @copy_gm_VectorLayout_to_l1_VectorLayout_half

//--- bf16.mlir

!gm = !tla.tensor<!tla.layout<!tla.shape<128>, !tla.stride<1>, !tla.shape<128>, VectorLayout>, !tla.coord<0>, !tla.ptr<bf16, gm, 2>>
!l1 = !tla.tensor<!tla.layout<!tla.shape<128>, !tla.stride<1>, !tla.shape<128>, VectorLayout>, !tla.coord<0>, !tla.ptr<bf16, l1, 512>>

module {
  "tla.func"() ({
  ^bb0(%arg0: !gm):
    %0 = "tla.alloc_ptr"() {size_bytes = 256 : i64} : () -> !tla.ptr<bf16, l1, 512>
    %1 = "tla.make_tensor_like"(%0, %arg0) {layoutTag = #tla.layout_tag<VectorLayout>} : (!tla.ptr<bf16, l1, 512>, !gm) -> !l1
    "tla.cube"() ({
      "tla.copy"(%1, %arg0) : (!l1, !gm) -> ()
    }) : () -> ()
    "tla.return"() : () -> ()
  }) {function_type = (!gm) -> (), sym_name = "vector_flat_bf16"} : () -> ()
}

// BF16-DAG: func.func private @copy_gm_VectorLayout_to_l1_VectorLayout_bf16
// BF16: call @copy_gm_VectorLayout_to_l1_VectorLayout_bf16
