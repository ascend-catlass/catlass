// RUN: %tla_compile %s --mlir-print-ir-after=tla-vector-region-simd -o %t 2>&1 | %filecheck %s --check-prefix=SIMD
// RUN: %filecheck %s --check-prefix=UNSIGNED < %t
// RUN: sed 's/64xui32/64xi32/g; s/ptr<ui32, ub, 4>/ptr<i32, ub, 4>/g; s/dtype = ui32/dtype = i32/g' %s | %tla_compile - --mlir-print-ir-after=tla-vector-region-simd -o %t.s 2>&1 | %filecheck %s --check-prefix=SIMD
// RUN: %filecheck %s --check-prefix=SIGNED < %t.s

!data = !tla.tensor<!tla.layout<!tla.shape<64>, !tla.stride<1>, !tla.shape<64>, RowMajor>, !tla.coord<0>, !tla.ptr<ui32, ub, 4>>

module {
  func.func @mull_pair(
      %a: memref<64xui32, #hivm.address_space<ub>>,
      %b: memref<64xui32, #hivm.address_space<ub>>,
      %lo_out: memref<64xui32, #hivm.address_space<ub>>,
      %hi_out: memref<64xui32, #hivm.address_space<ub>>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c64 = arith.constant 64 : index
    %a_tile = tla.tensor_desc %a shape [%c1, %c64, %c1, %c1] stride [%c64, %c1, %c1, %c1] origin_shape [%c1, %c64] coord [%c0, %c0] : memref<64xui32, #hivm.address_space<ub>> -> !data
    %b_tile = tla.tensor_desc %b shape [%c1, %c64, %c1, %c1] stride [%c64, %c1, %c1, %c1] origin_shape [%c1, %c64] coord [%c0, %c0] : memref<64xui32, #hivm.address_space<ub>> -> !data
    %lo_tile = tla.tensor_desc %lo_out shape [%c1, %c64, %c1, %c1] stride [%c64, %c1, %c1, %c1] origin_shape [%c1, %c64] coord [%c0, %c0] : memref<64xui32, #hivm.address_space<ub>> -> !data
    %hi_tile = tla.tensor_desc %hi_out shape [%c1, %c64, %c1, %c1] stride [%c64, %c1, %c1, %c1] origin_shape [%c1, %c64] coord [%c0, %c0] : memref<64xui32, #hivm.address_space<ub>> -> !data
    "tla.vec.func"() ({
      %x = tla.load %a_tile : !data -> !tla.vector<64xui32>
      %y = tla.load %b_tile : !data -> !tla.vector<64xui32>
      %active = "tla.create_mask"() {dtype = ui32, pattern = "ALL"} : () -> !tla.mask<64>
      %lo, %hi = "tla.mull"(%x, %y, %active) : (!tla.vector<64xui32>, !tla.vector<64xui32>, !tla.mask<64>) -> (!tla.vector<64xui32>, !tla.vector<64xui32>)
      tla.store %lo_tile, %lo : !data, !tla.vector<64xui32>
      tla.store %hi_tile, %hi : !data, !tla.vector<64xui32>
    }) : () -> ()
    return
  }
}

// SIMD-LABEL: func.func private @vector_region_
// SIMD: ave.hir.mull
// UNSIGNED: hivm.vmull.v64u32
// SIGNED: hivm.vmull.v64s32
