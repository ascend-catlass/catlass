// RUN: %tla_compile %s -o - --mlir-print-ir-after=tla-lower-scalar-ops 2>&1 | %filecheck %s

module {
  func.func @fast_div32(%d: i32) -> i32 {
    %magic, %shift = tla.scalar_fast_div_divisor %d : i32 -> i32, i32
    %r = arith.addi %magic, %shift : i32
    return %r : i32
  }

  func.func @fast_div64(%d: i64) -> i64 {
    %magic, %shift = tla.scalar_fast_div_divisor %d : i64 -> i64, i64
    %r = arith.addi %magic, %shift : i64
    return %r : i64
  }
}

// The width selects the helpers; shift is computed first and magic consumes
// both (divisor, shift), so neither result needs a struct return.
// CHECK-LABEL: func.func @fast_div32
// CHECK: %[[S32:.*]] = call @_mlir_ciface_tla_fast_div_shift_32(%arg0) : (i32) -> i32
// CHECK: %[[M32:.*]] = call @_mlir_ciface_tla_fast_div_magic_32(%arg0, %[[S32]]) : (i32, i32) -> i32
// CHECK: arith.addi %[[M32]], %[[S32]]
// CHECK-LABEL: func.func @fast_div64
// CHECK: %[[S64:.*]] = call @_mlir_ciface_tla_fast_div_shift_64(%arg0) : (i64) -> i64
// CHECK: %[[M64:.*]] = call @_mlir_ciface_tla_fast_div_magic_64(%arg0, %[[S64]]) : (i64, i64) -> i64
// CHECK: arith.addi %[[M64]], %[[S64]]
// CHECK-NOT: tla.scalar_fast_div_divisor
