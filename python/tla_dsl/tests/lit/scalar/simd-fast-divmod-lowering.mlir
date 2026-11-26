// RUN: %tla_compile %s -o - --mlir-print-ir-after=tla-lower-scalar-ops 2>&1 | %filecheck %s

module {
  func.func @div32(%x: i32, %m: i32, %s: i32) -> i32 {
    %q = tla.scalar_simd_fast_div %x, %m, %s : i32, i32, i32 -> i32
    return %q : i32
  }

  func.func @mod32(%x: i32, %d: i32, %m: i32, %s: i32) -> i32 {
    %r = tla.scalar_simd_fast_mod %x, %d, %m, %s : i32, i32, i32, i32 -> i32
    return %r : i32
  }

  func.func @div64(%x: i64, %m: i64, %s: i64) -> i64 {
    %q = tla.scalar_simd_fast_div %x, %m, %s : i64, i64, i64 -> i64
    return %q : i64
  }

  func.func @mod64(%x: i64, %d: i64, %m: i64, %s: i64) -> i64 {
    %r = tla.scalar_simd_fast_mod %x, %d, %m, %s : i64, i64, i64, i64 -> i64
    return %r : i64
  }
}

// Each op becomes one scalar-return helper call; the width selects the helper.
// CHECK-LABEL: func.func @div32
// CHECK: %[[D32:.*]] = call @_mlir_ciface_tla_simd_fast_div_32(%arg0, %arg1, %arg2) : (i32, i32, i32) -> i32
// CHECK-LABEL: func.func @mod32
// CHECK: %[[M32:.*]] = call @_mlir_ciface_tla_simd_fast_mod_32(%arg0, %arg1, %arg2, %arg3) : (i32, i32, i32, i32) -> i32
// CHECK-LABEL: func.func @div64
// CHECK: %[[D64:.*]] = call @_mlir_ciface_tla_simd_fast_div_64(%arg0, %arg1, %arg2) : (i64, i64, i64) -> i64
// CHECK-LABEL: func.func @mod64
// CHECK: %[[M64:.*]] = call @_mlir_ciface_tla_simd_fast_mod_64(%arg0, %arg1, %arg2, %arg3) : (i64, i64, i64, i64) -> i64
// CHECK-NOT: tla.scalar_simd_fast_div
// CHECK-NOT: tla.scalar_simd_fast_mod
