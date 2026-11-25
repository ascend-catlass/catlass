// RUN: %tla_compile %s -o - --mlir-print-ir-after=tla-lower-scalar-ops 2>&1 | %filecheck %s

module {
  func.func @bit_queries(%x: i64) -> i64 {
    %a = tla.scalar_bit_op "bcnt0" %x : i64 to i64
    %b = tla.scalar_bit_op "bcnt1" %x : i64 to i64
    %c = tla.scalar_bit_op "clz" %x : i64 to i64
    %d = tla.scalar_bit_op "sff0" %x : i64 to i64
    %e = tla.scalar_bit_op "sff1" %x : i64 to i64
    %ab = arith.addi %a, %b : i64
    %cd = arith.addi %c, %d : i64
    %abc = arith.addi %ab, %cd : i64
    %r = arith.addi %abc, %e : i64
    return %r : i64
  }
}

// Each query rewrites to the meta-op bitcode helper with the same mode name.
// CHECK-LABEL: func.func @bit_queries
// CHECK: %[[BCNT0:.*]] = call @_mlir_ciface_bcnt0(%arg0) : (i64) -> i64
// CHECK: %[[BCNT1:.*]] = call @_mlir_ciface_bcnt1(%arg0) : (i64) -> i64
// CHECK: %[[CLZ:.*]] = call @_mlir_ciface_clz(%arg0) : (i64) -> i64
// CHECK: %[[SFF0:.*]] = call @_mlir_ciface_sff0(%arg0) : (i64) -> i64
// CHECK: %[[SFF1:.*]] = call @_mlir_ciface_sff1(%arg0) : (i64) -> i64
// CHECK: arith.addi %[[BCNT0]], %[[BCNT1]]
// CHECK-NOT: tla.scalar_bit_op
