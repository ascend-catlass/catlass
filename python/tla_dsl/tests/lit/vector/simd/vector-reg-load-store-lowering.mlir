// RUN: %tla_compile %s --mlir-print-ir-after=tla-vector-region-simd -o %t 2>&1 | %filecheck %s --check-prefix=WORD
// RUN: sed 's/i64/ui64/g' %s | %tla_compile - --mlir-print-ir-after=tla-vector-region-simd -o %t 2>&1 | %filecheck %s --check-prefix=WORD
// RUN: sed 's/i64/f64/g' %s | %tla_compile - --mlir-print-ir-after=tla-vector-region-simd -o %t 2>&1 | %filecheck %s --check-prefix=FLOAT
// RUN: sed 's/!tla.vector<32xi64>/!tla.vector<32xui64>/g' %s | not %tla_compile - -o %t 2>&1 | %filecheck %s --check-prefix=TYPE-MISMATCH

!tile = !tla.tensor<!tla.layout<!tla.shape<32>, !tla.stride<1>, !tla.shape<32>, RowMajor>, !tla.coord<0>, !tla.ptr<i64, ub, 256>>

module {
  func.func @load_store(
      %src_memref: memref<32xi64, #hivm.address_space<ub>>,
      %dst_memref: memref<32xi64, #hivm.address_space<ub>>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c32 = arith.constant 32 : index
    %src = tla.tensor_desc %src_memref shape [%c1, %c32, %c1, %c1] stride [%c32, %c1, %c1, %c1] origin_shape [%c1, %c32] coord [%c0, %c0] : memref<32xi64, #hivm.address_space<ub>> -> !tile
    %dst = tla.tensor_desc %dst_memref shape [%c1, %c32, %c1, %c1] stride [%c32, %c1, %c1, %c1] origin_shape [%c1, %c32] coord [%c0, %c0] : memref<32xi64, #hivm.address_space<ub>> -> !tile
    "tla.vec.func"() ({
      %v = tla.load %src : !tile -> !tla.vector<32xi64>
      tla.store %dst, %v : !tile, !tla.vector<32xi64>
    }) : () -> ()
    return
  }
}

// WORD-LABEL: func.func private @vector_region_
// WORD: ave.hir.vload <NORM>
// WORD-SAME: into vector<64xi32>
// WORD: vector.bitcast
// WORD-SAME: to vector<32xi64>
// WORD: vector.bitcast
// WORD-SAME: to vector<64xi32>
// WORD: ave.hir.masked_store
// WORD-SAME: vector<64xi32>

// FLOAT-LABEL: func.func private @vector_region_
// FLOAT: ave.hir.vload <NORM>
// FLOAT-SAME: into vector<64xi32>
// FLOAT: vector.bitcast
// FLOAT-SAME: to vector<32xf64>
// FLOAT: vector.bitcast
// FLOAT-SAME: to vector<64xi32>
// FLOAT: ave.hir.masked_store
// FLOAT-SAME: vector<64xi32>

// TYPE-MISMATCH: 64-bit register load requires matching element types
