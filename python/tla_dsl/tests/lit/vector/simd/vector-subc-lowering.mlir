// RUN: %tla_compile %s --mlir-print-ir-after=tla-vector-region-simd -o %t 2>&1 | %filecheck %s -D'$carry_op=tla_subc_u32'
// RUN: sed 's/64xui32/64xi32/g; s/ptr<ui32, ub, 4>/ptr<i32, ub, 4>/g; s/dtype = ui32/dtype = i32/g' %s | %tla_compile - --mlir-print-ir-after=tla-vector-region-simd -o %t.s 2>&1 | %filecheck %s -D'$carry_op=tla_subc_i32'

!data = !tla.tensor<!tla.layout<!tla.shape<64>, !tla.stride<1>, !tla.shape<64>, RowMajor>, !tla.coord<0>, !tla.ptr<ui32, ub, 4>>

module {
  func.func @subc_chain(
      %a: memref<64xui32, #hivm.address_space<ub>>,
      %b: memref<64xui32, #hivm.address_space<ub>>,
      %out: memref<64xui32, #hivm.address_space<ub>>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c64 = arith.constant 64 : index
    %a_tile = tla.tensor_desc %a shape [%c1, %c64, %c1, %c1] stride [%c64, %c1, %c1, %c1] origin_shape [%c1, %c64] coord [%c0, %c0] : memref<64xui32, #hivm.address_space<ub>> -> !data
    %b_tile = tla.tensor_desc %b shape [%c1, %c64, %c1, %c1] stride [%c64, %c1, %c1, %c1] origin_shape [%c1, %c64] coord [%c0, %c0] : memref<64xui32, #hivm.address_space<ub>> -> !data
    %out_tile = tla.tensor_desc %out shape [%c1, %c64, %c1, %c1] stride [%c64, %c1, %c1, %c1] origin_shape [%c1, %c64] coord [%c0, %c0] : memref<64xui32, #hivm.address_space<ub>> -> !data
    "tla.vec.func"() ({
      %x = tla.load %a_tile : !data -> !tla.vector<64xui32>
      %y = tla.load %b_tile : !data -> !tla.vector<64xui32>
      %cin = "tla.create_mask"() {dtype = ui32, pattern = "ALL"} : () -> !tla.mask<64>
      %active = "tla.create_mask"() {dtype = ui32, pattern = "ALL"} : () -> !tla.mask<64>
      %diff, %carry = "tla.subc"(%x, %y, %cin, %active) : (!tla.vector<64xui32>, !tla.vector<64xui32>, !tla.mask<64>, !tla.mask<64>) -> (!tla.vector<64xui32>, !tla.mask<64>)
      %diff2, %unused = "tla.subc"(%diff, %y, %carry, %active) : (!tla.vector<64xui32>, !tla.vector<64xui32>, !tla.mask<64>, !tla.mask<64>) -> (!tla.vector<64xui32>, !tla.mask<64>)
      tla.store %out_tile, %diff2 : !data, !tla.vector<64xui32>
    }) : () -> ()
    return
  }
}

// CHECK-DAG: func.func private @[[$carry_op]](vector<64xi32>, vector<64xi32>, vector<256xi1>, vector<256xi1>, !llvm.ptr<6>) -> vector<64xi32>
// CHECK-DAG: func.func private @tla_b32_read_carry(!llvm.ptr<6>) -> vector<256xi1>
// CHECK-LABEL: func.func private @vector_region_
// CHECK-COUNT-2: llvm.alloca {{.*}} x vector<256xi1> {alignment = 32 : i64}
// CHECK: %[[PARTIAL:[a-zA-Z0-9_]+]] = call @[[$carry_op]]({{.*}}, %[[SCRATCH:[a-zA-Z0-9_]+]])
// CHECK: %[[CARRY:[a-zA-Z0-9_]+]] = call @tla_b32_read_carry(%[[SCRATCH]])
// CHECK: call @[[$carry_op]](%[[PARTIAL]], {{.*}}, %[[CARRY]],
// CHECK-NOT: call @tla_b32_read_carry
// CHECK: return
