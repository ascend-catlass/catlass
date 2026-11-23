// RUN: %tla_compile %s -o - | %filecheck %s
!gm = !tla.ptr<i32, gm, 4>
!byte = !tla.ptr<i8, gm, 4>
!view = !tla.tensor<!tla.layout<!tla.shape<16>, !tla.stride<1>, !tla.shape<64>, RowMajor>, !tla.coord<32>, !gm>
!buffer = memref<64xi32, strided<[1], offset: 3>, #hivm.address_space<gm>>
module {
  func.func @address(%buf: !buffer) {
    %z = arith.constant 0 : index
    %one = arith.constant 1 : index
    %two = arith.constant 2 : index
    %size = arith.constant 16 : index
    %full = arith.constant 64 : index
    %origin = arith.constant 32 : index
    %delta = arith.constant -4 : i64
    %v = tla.tensor_desc %buf shape [%one, %size, %one, %one] stride [%full, %one, %one, %one] origin_shape [%one, %full] coord [%z, %origin] : !buffer -> !view
    "tla.vector"() ({
      %p = "tla.tensor_ptr"(%v) : (!view) -> !gm
      %ip = tla.ptr_add %p, %two : !gm, index -> !gm
      %wide = tla.recast_ptr %ip : !gm -> !byte
      tla.arch.dcache_prefetch %wide, %delta : !byte
      %bp = tla.recast_ptr %p : !gm -> !byte
      %narrow = tla.ptr_add %bp, %two : !byte, index -> !byte
      tla.arch.dcache_prefetch %narrow, %delta : !byte
    }) : () -> ()
    return
  }
func.func @merged(%cond: i1, %lhs: i64, %rhs: i64) {
  "tla.vector"() ({
    %a = tla.inttoptr %lhs : i64 -> !gm
    %b = tla.inttoptr %rhs : i64 -> !gm
    %p = scf.if %cond -> (!gm) {
      scf.yield %a : !gm
    } else {
      scf.yield %b : !gm
    }
    %zero = arith.constant 0 : i64
    tla.arch.dcache_prefetch %p, %zero : !gm
  }) : () -> ()
  return
}

}
// CHECK-LABEL: func.func @address
// tensor_ptr supplies the aligned backing pointer, independent of view coordinates.
// CHECK-DAG: %[[TWO:.*]] = llvm.mlir.constant(2 : i64)
// CHECK-DAG: %[[EIGHT:.*]] = llvm.mlir.constant(8 : i64)
// CHECK-DAG: %[[DELTA:.*]] = llvm.mlir.constant(-4 : i64)
// CHECK: %[[PTR:.*]] = memref.extract_aligned_pointer_as_index %arg0
// CHECK: %[[BASE:.*]] = builtin.unrealized_conversion_cast %[[PTR]] : index to i64
// CHECK: %[[WIDE:.*]] = llvm.add %[[BASE]], %[[EIGHT]]
// CHECK: call @_mlir_ciface_tla_dcache_prefetch_aiv(%[[WIDE]], %[[DELTA]])
// CHECK: %[[BYTE:.*]] = llvm.add %[[BASE]], %[[TWO]]
// CHECK: call @_mlir_ciface_tla_dcache_prefetch_aiv(%[[BYTE]], %[[DELTA]])
// CHECK-LABEL: func.func @merged
// CHECK: call @_mlir_ciface_tla_dcache_prefetch_aiv
