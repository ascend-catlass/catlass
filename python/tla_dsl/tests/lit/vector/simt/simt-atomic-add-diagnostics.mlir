// One error case per invocation: the verifier checks ops in parallel and may
// stop emitting diagnostics after the first failure, so one invocation is not
// guaranteed to report every bad op. Each block is extracted into its own
// file and compiled separately, which keeps each CHECK deterministic.
//
// The pointer comes from an integer address so each block is just the op under
// test: these cases are all rejected by the op verifier, before any pass reads
// the pointer chain.

// RUN: sed -n '/^\/\/ BEGIN_ELEMENT$/,/^\/\/ END_ELEMENT$/p' %s | sed '1d;$d;s|^// ||' > %t.ELEMENT.mlir && not %tla_compile %t.ELEMENT.mlir -o %t 2>&1 | %filecheck %s --check-prefix=ELEMENT
// ELEMENT: error:
// ELEMENT-SAME: unsupported atomic add element type 'f64'
// BEGIN_ELEMENT
// module {
//   func.func @unsupported_element_type(%addr: i64) {
//     "tla.vec.func"() ({
//       %v = arith.constant 1.000000e+00 : f64
//       %p = tla.inttoptr %addr : i64 -> !tla.ptr<f64, gm, 8>
//       tla.simt_atomic_add %p, %v : !tla.ptr<f64, gm, 8>, f64 -> f64
//     }) {mode = "simt", thread_block_dim = array<i64: 64, 1, 1>} : () -> ()
//     return
//   }
// }
// END_ELEMENT

// RUN: sed -n '/^\/\/ BEGIN_UB64$/,/^\/\/ END_UB64$/p' %s | sed '1d;$d;s|^// ||' > %t.UB64.mlir && not %tla_compile %t.UB64.mlir -o %t 2>&1 | %filecheck %s --check-prefix=UB64
// UB64: error:
// UB64-SAME: 64-bit atomic add is only supported on gm pointers
// BEGIN_UB64
// module {
//   func.func @i64_on_ub(%addr: i64) {
//     "tla.vec.func"() ({
//       %v = arith.constant 1 : i64
//       %p = tla.inttoptr %addr : i64 -> !tla.ptr<i64, ub, 8>
//       tla.simt_atomic_add %p, %v : !tla.ptr<i64, ub, 8>, i64 -> i64
//     }) {mode = "simt", thread_block_dim = array<i64: 64, 1, 1>} : () -> ()
//     return
//   }
// }
// END_UB64

// RUN: sed -n '/^\/\/ BEGIN_VALUE$/,/^\/\/ END_VALUE$/p' %s | sed '1d;$d;s|^// ||' > %t.VALUE.mlir && not %tla_compile %t.VALUE.mlir -o %t 2>&1 | %filecheck %s --check-prefix=VALUE
// VALUE: error:
// VALUE-SAME: value type 'f32' must match the pointer's pointee type 'i32'
// BEGIN_VALUE
// module {
//   func.func @value_type_mismatch(%addr: i64) {
//     "tla.vec.func"() ({
//       %v = arith.constant 1.000000e+00 : f32
//       %p = tla.inttoptr %addr : i64 -> !tla.ptr<i32, gm, 4>
//       tla.simt_atomic_add %p, %v : !tla.ptr<i32, gm, 4>, f32 -> f32
//     }) {mode = "simt", thread_block_dim = array<i64: 64, 1, 1>} : () -> ()
//     return
//   }
// }
// END_VALUE

// RUN: sed -n '/^\/\/ BEGIN_RESULT$/,/^\/\/ END_RESULT$/p' %s | sed '1d;$d;s|^// ||' > %t.RESULT.mlir && not %tla_compile %t.RESULT.mlir -o %t 2>&1 | %filecheck %s --check-prefix=RESULT
// RESULT: error:
// RESULT-SAME: result type 'f32' must match the pointer's pointee type 'i32'
// BEGIN_RESULT
// module {
//   func.func @result_type_mismatch(%addr: i64) {
//     "tla.vec.func"() ({
//       %v = arith.constant 1 : i32
//       %p = tla.inttoptr %addr : i64 -> !tla.ptr<i32, gm, 4>
//       tla.simt_atomic_add %p, %v : !tla.ptr<i32, gm, 4>, i32 -> f32
//     }) {mode = "simt", thread_block_dim = array<i64: 64, 1, 1>} : () -> ()
//     return
//   }
// }
// END_RESULT
