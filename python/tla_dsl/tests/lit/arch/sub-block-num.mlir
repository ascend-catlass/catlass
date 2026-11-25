// RUN: split-file %s %t
// RUN: %tla_compile %t/sub-block-count-mixed.mlir -o %t/output --mlir-print-ir-before=tla-split-mixed-func 2>&1 | %filecheck %t/sub-block-count-mixed.mlir --check-prefix=BEFORE
// RUN: %tla_compile %t/sub-block-count-mixed.mlir -o %t/output --mlir-print-ir-after=tla-split-mixed-func 2>&1 | %filecheck %t/sub-block-count-mixed.mlir --check-prefix=SPLIT
// RUN: %tla_compile %t/sub-block-count-mixed.mlir -o %t/output --mlir-print-ir-after=tla-lower-block-idx 2>&1 | %filecheck %t/sub-block-count-mixed.mlir --check-prefix=LOWER
// RUN: %tla_compile %t/sub-block-count-local.mlir -o %t/output --mlir-print-ir-after=tla-lower-block-idx 2>&1 | %filecheck %t/sub-block-count-local.mlir
// RUN: not %tla_compile %t/sub-block-count-core-diagnostics.mlir -o - 2>&1 | %filecheck %t/sub-block-count-core-diagnostics.mlir
// RUN: not %tla_compile %t/sub-block-count-simd-diagnostics.mlir -o - 2>&1 | %filecheck %t/sub-block-count-simd-diagnostics.mlir
// RUN: not %tla_compile %t/sub-block-count-simt-diagnostics.mlir -o - 2>&1 | %filecheck %t/sub-block-count-simt-diagnostics.mlir

//--- sub-block-count-mixed.mlir
// Shared outer SSA is cloned for both cores, then lowered per core.

module {
  tla.func @query(%cube: memref<1xi64>, %vector: memref<1xi64>) {
    %i = arith.constant 0 : index
    %n = tla.arch.sub_block_num -> i64
    "tla.cube"() ({
      memref.store %n, %cube[%i] : memref<1xi64>
    }) : () -> ()
    "tla.vector"() ({
      memref.store %n, %vector[%i] : memref<1xi64>
    }) : () -> ()
    tla.return
  }
}

// BEFORE-LABEL: func.func @query(
// BEFORE: %[[N:.*]] = tla.arch.sub_block_num
// BEFORE: memref.store %[[N]],
// BEFORE: memref.store %[[N]],

// SPLIT-LABEL: func.func @query_mix_aic(
// SPLIT: %[[N:.*]] = tla.arch.sub_block_num
// SPLIT: memref.store %[[N]],

// SPLIT-LABEL: func.func @query_mix_aiv(
// SPLIT: %[[N:.*]] = tla.arch.sub_block_num
// SPLIT: memref.store %[[N]],

// LOWER-LABEL: func.func @query_mix_aic(
// LOWER: %[[N:.*]] = arith.constant 1 : i64
// LOWER: memref.store %[[N]],

// LOWER-LABEL: func.func @query_mix_aiv(
// LOWER: %[[N:.*]] = hivm.hir.get_sub_block_num
// LOWER: memref.store %[[N]],

//--- sub-block-count-local.mlir
// Queries inside core scopes and an outer value used only by vector.

module {
  tla.func @scope_local(%a: memref<2xi64>, %v: memref<2xi64>) {
    %zero = arith.constant 0 : index
    %one = arith.constant 1 : index
    %vector_only = tla.arch.sub_block_num -> i64
    "tla.cube"() ({
      %n = tla.arch.sub_block_num -> i64
      memref.store %n, %a[%zero] : memref<2xi64>
    }) : () -> ()
    "tla.vector"() ({
      %first = tla.arch.sub_block_num -> i64
      %second = tla.arch.sub_block_num -> i64
      %sum = arith.addi %first, %second : i64
      memref.store %sum, %v[%zero] : memref<2xi64>
      memref.store %vector_only, %v[%one] : memref<2xi64>
    }) : () -> ()
    tla.return
  }
}

// CHECK-LABEL: func.func @scope_local_mix_aic(
// CHECK: arith.constant 1 : i64
// CHECK: memref.store
// CHECK-NOT: hivm.hir.get_sub_block_num
// CHECK-LABEL: func.func @scope_local_mix_aiv(
// CHECK: hivm.hir.get_sub_block_num
// CHECK: arith.addi
// CHECK: memref.store
// CHECK: memref.store

//--- sub-block-count-core-diagnostics.mlir
// A query with no classified device core must fail.

module {
  func.func private @unclassified() -> i64 {
    %n = tla.arch.sub_block_num -> i64
    return %n : i64
  }
  func.func @entry(%out: memref<1xi64>) {
    %i = arith.constant 0 : index
    %n = func.call @unclassified() : () -> i64
    memref.store %n, %out[%i] : memref<1xi64>
    return
  }
}

// CHECK: 'tla.arch.sub_block_num' op requires an enclosing device function with AIC or AIV core type

//--- sub-block-count-simd-diagnostics.mlir
// A direct SIMD vec.func query must fail even when unused.

module {
  func.func @illegal() {
    "tla.vector"() ({
      "tla.vec.func"() ({
        %unused = tla.arch.sub_block_num -> i64
      }) {mode = "simd"} : () -> ()
    }) : () -> ()
    return
  }
}

// CHECK: compute outside tla.vec.func and pass the scalar into the region

//--- sub-block-count-simt-diagnostics.mlir
// A direct SIMT vec.func query must fail even when unused.

module {
  func.func @illegal() {
    "tla.vector"() ({
      "tla.vec.func"() ({
        %unused = tla.arch.sub_block_num -> i64
      }) {mode = "simt", thread_block_dim = array<i64: 32, 1, 1>} : () -> ()
    }) : () -> ()
    return
  }
}

// CHECK: compute outside tla.vec.func and pass the scalar into the region
