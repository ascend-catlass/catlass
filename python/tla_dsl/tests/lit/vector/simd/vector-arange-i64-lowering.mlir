// RUN: %tla_compile %s -o - --mlir-print-ir-after=tla-vector-region-simd 2>&1 | %filecheck %s --implicit-check-not=ave.hir.vci

// CHECK-DAG: func.func private @arange_int64_increase(i64) -> vector<32xi64>
// CHECK-DAG: func.func private @arange_int64_decrease(i64) -> vector<32xi64>
// CHECK: call @arange_int64_increase(%{{.*}}) : (i64) -> vector<32xi64>
// CHECK: call @arange_int64_decrease(%{{.*}}) : (i64) -> vector<32xi64>

!out = !tla.tensor<!tla.layout<!tla.shape<32>, !tla.stride<1>, !tla.shape<32>, RowMajor>, !tla.coord<0>, !tla.ptr<i64, ub, 32>>

func.func @arange_i64(
    %inc_dst: memref<32xi64, #hivm.address_space<ub>>,
    %dec_dst: memref<32xi64, #hivm.address_space<ub>>,
    %base: i64) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c32 = arith.constant 32 : index
  %inc = tla.tensor_desc %inc_dst shape [%c1, %c32, %c1, %c1] stride [%c32, %c1, %c1, %c1] origin_shape [%c1, %c32] coord [%c0, %c0] : memref<32xi64, #hivm.address_space<ub>> -> !out
  %dec = tla.tensor_desc %dec_dst shape [%c1, %c32, %c1, %c1] stride [%c32, %c1, %c1, %c1] origin_shape [%c1, %c32] coord [%c0, %c0] : memref<32xi64, #hivm.address_space<ub>> -> !out
  "tla.vec.func"() ({
    %literal = arith.constant 4294967290 : i64
    %ascending = tla.arange %literal {order = "increase"} : i64 -> !tla.vector<32xi64>
    %descending = tla.arange %base {order = "decrease"} : i64 -> !tla.vector<32xi64>
    tla.store %inc, %ascending : !out, !tla.vector<32xi64>
    tla.store %dec, %descending : !out, !tla.vector<32xi64>
  }) {mode = "simd"} : () -> ()
  return
}
