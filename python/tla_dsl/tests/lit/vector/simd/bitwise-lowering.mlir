// RUN: %tla_compile %s -o - --mlir-print-ir-after=tla-vector-region-simd 2>&1 | %filecheck %s

// CHECK: ave.hir.preg.not
// CHECK: ave.hir.preg.not
// CHECK: ave.hir.preg.and
// CHECK: ave.hir.preg.and
// CHECK: ave.hir.preg.or
// CHECK: ave.hir.preg.or
// CHECK: ave.hir.preg.xor
// CHECK: ave.hir.preg.xor
// CHECK: ave.hir.vnot
// CHECK: ave.hir.vand
// CHECK: ave.hir.vand
// CHECK: ave.hir.vor
// CHECK: ave.hir.vor
// CHECK: ave.hir.vxor
// CHECK: ave.hir.vxor
// CHECK-NOT: tla.bitwise_not
// CHECK-NOT: tla.bitwise_and
// CHECK-NOT: tla.bitwise_or
// CHECK-NOT: tla.bitwise_xor

!vec = !tla.tensor<!tla.layout<!tla.shape<64>, !tla.stride<1>, !tla.shape<64>, RowMajor>, !tla.coord<0>, !tla.ptr<i32, ub, 4>>
!vec_alt = !tla.tensor<!tla.layout<!tla.shape<64>, !tla.stride<1>, !tla.shape<64>, RowMajor>, !tla.coord<1>, !tla.ptr<i32, ub, 4>>

module {
  func.func @bitwise_lowering(
      %src0_memref: memref<64xi32, #hivm.address_space<ub>>,
      %src1_memref: memref<64xi32, #hivm.address_space<ub>>,
      %dst_mask_not_memref: memref<64xi32, #hivm.address_space<ub>>,
      %dst_mask_and_memref: memref<64xi32, #hivm.address_space<ub>>,
      %dst_mask_or_memref: memref<64xi32, #hivm.address_space<ub>>,
      %dst_mask_xor_memref: memref<64xi32, #hivm.address_space<ub>>,
      %dst_reg_not_memref: memref<64xi32, #hivm.address_space<ub>>,
      %dst_reg_and_memref: memref<64xi32, #hivm.address_space<ub>>,
      %dst_reg_or_memref: memref<64xi32, #hivm.address_space<ub>>,
      %dst_reg_xor_memref: memref<64xi32, #hivm.address_space<ub>>) {
    %src0_c0 = arith.constant 0 : index
    %src0_c1 = arith.constant 1 : index
    %src0_c64 = arith.constant 64 : index
    %src0 = tla.tensor_desc %src0_memref shape [%src0_c1, %src0_c64, %src0_c1, %src0_c1] stride [%src0_c64, %src0_c1, %src0_c1, %src0_c1] origin_shape [%src0_c1, %src0_c64] coord [%src0_c0, %src0_c0] : memref<64xi32, #hivm.address_space<ub>> -> !vec
    %src1_c0 = arith.constant 0 : index
    %src1_c1 = arith.constant 1 : index
    %src1_c64 = arith.constant 64 : index
    %src1 = tla.tensor_desc %src1_memref shape [%src1_c1, %src1_c64, %src1_c1, %src1_c1] stride [%src1_c64, %src1_c1, %src1_c1, %src1_c1] origin_shape [%src1_c1, %src1_c64] coord [%src1_c0, %src1_c0] : memref<64xi32, #hivm.address_space<ub>> -> !vec
    %dst_mask_not_c0 = arith.constant 0 : index
    %dst_mask_not_c1 = arith.constant 1 : index
    %dst_mask_not_c64 = arith.constant 64 : index
    %dst_mask_not = tla.tensor_desc %dst_mask_not_memref shape [%dst_mask_not_c1, %dst_mask_not_c64, %dst_mask_not_c1, %dst_mask_not_c1] stride [%dst_mask_not_c64, %dst_mask_not_c1, %dst_mask_not_c1, %dst_mask_not_c1] origin_shape [%dst_mask_not_c1, %dst_mask_not_c64] coord [%dst_mask_not_c0, %dst_mask_not_c0] : memref<64xi32, #hivm.address_space<ub>> -> !vec
    %dst_mask_and_c0 = arith.constant 0 : index
    %dst_mask_and_c1 = arith.constant 1 : index
    %dst_mask_and_c64 = arith.constant 64 : index
    %dst_mask_and = tla.tensor_desc %dst_mask_and_memref shape [%dst_mask_and_c1, %dst_mask_and_c64, %dst_mask_and_c1, %dst_mask_and_c1] stride [%dst_mask_and_c64, %dst_mask_and_c1, %dst_mask_and_c1, %dst_mask_and_c1] origin_shape [%dst_mask_and_c1, %dst_mask_and_c64] coord [%dst_mask_and_c0, %dst_mask_and_c0] : memref<64xi32, #hivm.address_space<ub>> -> !vec
    %dst_mask_or_c0 = arith.constant 0 : index
    %dst_mask_or_c1 = arith.constant 1 : index
    %dst_mask_or_c64 = arith.constant 64 : index
    %dst_mask_or = tla.tensor_desc %dst_mask_or_memref shape [%dst_mask_or_c1, %dst_mask_or_c64, %dst_mask_or_c1, %dst_mask_or_c1] stride [%dst_mask_or_c64, %dst_mask_or_c1, %dst_mask_or_c1, %dst_mask_or_c1] origin_shape [%dst_mask_or_c1, %dst_mask_or_c64] coord [%dst_mask_or_c0, %dst_mask_or_c0] : memref<64xi32, #hivm.address_space<ub>> -> !vec
    %dst_mask_xor_c0 = arith.constant 0 : index
    %dst_mask_xor_c1 = arith.constant 1 : index
    %dst_mask_xor_c64 = arith.constant 64 : index
    %dst_mask_xor = tla.tensor_desc %dst_mask_xor_memref shape [%dst_mask_xor_c1, %dst_mask_xor_c64, %dst_mask_xor_c1, %dst_mask_xor_c1] stride [%dst_mask_xor_c64, %dst_mask_xor_c1, %dst_mask_xor_c1, %dst_mask_xor_c1] origin_shape [%dst_mask_xor_c1, %dst_mask_xor_c64] coord [%dst_mask_xor_c0, %dst_mask_xor_c0] : memref<64xi32, #hivm.address_space<ub>> -> !vec
    %dst_reg_not_c0 = arith.constant 0 : index
    %dst_reg_not_c1 = arith.constant 1 : index
    %dst_reg_not_c64 = arith.constant 64 : index
    %dst_reg_not = tla.tensor_desc %dst_reg_not_memref shape [%dst_reg_not_c1, %dst_reg_not_c64, %dst_reg_not_c1, %dst_reg_not_c1] stride [%dst_reg_not_c64, %dst_reg_not_c1, %dst_reg_not_c1, %dst_reg_not_c1] origin_shape [%dst_reg_not_c1, %dst_reg_not_c64] coord [%dst_reg_not_c0, %dst_reg_not_c0] : memref<64xi32, #hivm.address_space<ub>> -> !vec
    %dst_reg_and_c0 = arith.constant 0 : index
    %dst_reg_and_c1 = arith.constant 1 : index
    %dst_reg_and_c64 = arith.constant 64 : index
    %dst_reg_and = tla.tensor_desc %dst_reg_and_memref shape [%dst_reg_and_c1, %dst_reg_and_c64, %dst_reg_and_c1, %dst_reg_and_c1] stride [%dst_reg_and_c64, %dst_reg_and_c1, %dst_reg_and_c1, %dst_reg_and_c1] origin_shape [%dst_reg_and_c1, %dst_reg_and_c64] coord [%dst_reg_and_c0, %dst_reg_and_c0] : memref<64xi32, #hivm.address_space<ub>> -> !vec
    %dst_reg_or_c0 = arith.constant 0 : index
    %dst_reg_or_c1 = arith.constant 1 : index
    %dst_reg_or_c64 = arith.constant 64 : index
    %dst_reg_or = tla.tensor_desc %dst_reg_or_memref shape [%dst_reg_or_c1, %dst_reg_or_c64, %dst_reg_or_c1, %dst_reg_or_c1] stride [%dst_reg_or_c64, %dst_reg_or_c1, %dst_reg_or_c1, %dst_reg_or_c1] origin_shape [%dst_reg_or_c1, %dst_reg_or_c64] coord [%dst_reg_or_c0, %dst_reg_or_c0] : memref<64xi32, #hivm.address_space<ub>> -> !vec
    %dst_reg_xor_c0 = arith.constant 0 : index
    %dst_reg_xor_c1 = arith.constant 1 : index
    %dst_reg_xor_c64 = arith.constant 64 : index
    %dst_reg_xor = tla.tensor_desc %dst_reg_xor_memref shape [%dst_reg_xor_c1, %dst_reg_xor_c64, %dst_reg_xor_c1, %dst_reg_xor_c1] stride [%dst_reg_xor_c64, %dst_reg_xor_c1, %dst_reg_xor_c1, %dst_reg_xor_c1] origin_shape [%dst_reg_xor_c1, %dst_reg_xor_c64] coord [%dst_reg_xor_c0, %dst_reg_xor_c0] : memref<64xi32, #hivm.address_space<ub>> -> !vec
    "tla.vector"() ({
      "tla.vec.func"() ({
        %shape = "tla.make_shape"() : () -> !tla.shape<64>
        %coord = "tla.make_coord"() : () -> !tla.coord<0>
        %src0_tile = "tla.tile_view"(%src0, %shape, %coord) : (!vec, !tla.shape<64>, !tla.coord<0>) -> !vec
        %src1_tile = "tla.tile_view"(%src1, %shape, %coord) : (!vec, !tla.shape<64>, !tla.coord<0>) -> !vec
        %dst_mask_not_tile = "tla.tile_view"(%dst_mask_not, %shape, %coord) : (!vec, !tla.shape<64>, !tla.coord<0>) -> !vec
        %dst_mask_and_tile = "tla.tile_view"(%dst_mask_and, %shape, %coord) : (!vec, !tla.shape<64>, !tla.coord<0>) -> !vec
        %dst_mask_or_tile = "tla.tile_view"(%dst_mask_or, %shape, %coord) : (!vec, !tla.shape<64>, !tla.coord<0>) -> !vec
        %dst_mask_xor_tile = "tla.tile_view"(%dst_mask_xor, %shape, %coord) : (!vec, !tla.shape<64>, !tla.coord<0>) -> !vec
        %dst_reg_not_tile = "tla.tile_view"(%dst_reg_not, %shape, %coord) : (!vec, !tla.shape<64>, !tla.coord<0>) -> !vec
        %dst_reg_and_tile = "tla.tile_view"(%dst_reg_and, %shape, %coord) : (!vec, !tla.shape<64>, !tla.coord<0>) -> !vec
        %dst_reg_or_tile = "tla.tile_view"(%dst_reg_or, %shape, %coord) : (!vec, !tla.shape<64>, !tla.coord<0>) -> !vec
        %dst_reg_xor_tile = "tla.tile_view"(%dst_reg_xor, %shape, %coord) : (!vec, !tla.shape<64>, !tla.coord<0>) -> !vec
        %all = "tla.create_mask"() {pattern = "ALL", dtype = i32} : () -> !tla.mask<64>
        %h = "tla.create_mask"() {pattern = "H", dtype = i32} : () -> !tla.mask<64>
        %q = "tla.create_mask"() {pattern = "Q", dtype = i32} : () -> !tla.mask<64>
        %m4 = "tla.create_mask"() {pattern = "M4", dtype = i32} : () -> !tla.mask<64>
        %mask_bitwise_not_all = "tla.bitwise_not"(%q) : (!tla.mask<64>) -> !tla.mask<64>
        %mask_bitwise_not = "tla.bitwise_not"(%mask_bitwise_not_all, %all) : (!tla.mask<64>, !tla.mask<64>) -> !tla.mask<64>
        %mask_bitwise_and_all = "tla.bitwise_and"(%h, %m4) : (!tla.mask<64>, !tla.mask<64>) -> !tla.mask<64>
        %mask_bitwise_and = "tla.bitwise_and"(%mask_bitwise_and_all, %m4, %all) : (!tla.mask<64>, !tla.mask<64>, !tla.mask<64>) -> !tla.mask<64>
        %mask_bitwise_or_all = "tla.bitwise_or"(%q, %m4) : (!tla.mask<64>, !tla.mask<64>) -> !tla.mask<64>
        %mask_bitwise_or = "tla.bitwise_or"(%mask_bitwise_or_all, %m4, %all) : (!tla.mask<64>, !tla.mask<64>, !tla.mask<64>) -> !tla.mask<64>
        %mask_bitwise_xor_all = "tla.bitwise_xor"(%h, %m4) : (!tla.mask<64>, !tla.mask<64>) -> !tla.mask<64>
        %mask_bitwise_xor = "tla.bitwise_xor"(%mask_bitwise_xor_all, %m4, %all) : (!tla.mask<64>, !tla.mask<64>, !tla.mask<64>) -> !tla.mask<64>
        %reg0 = "tla.load"(%src0_tile) <{operandSegmentSizes = array<i32: 1, 0, 0>}> : (!vec) -> !tla.vector<64xi32>
        %reg1 = "tla.load"(%src1_tile) <{operandSegmentSizes = array<i32: 1, 0, 0>}> : (!vec) -> !tla.vector<64xi32>
        %zero = arith.constant 0 : i32
        %zero_reg = "tla.full"(%zero) : (i32) -> !tla.vector<64xi32>
        %out_mask_not = "tla.where"(%mask_bitwise_not, %reg0, %zero_reg) : (!tla.mask<64>, !tla.vector<64xi32>, !tla.vector<64xi32>) -> !tla.vector<64xi32>
        %out_mask_and = "tla.where"(%mask_bitwise_and, %reg0, %zero_reg) : (!tla.mask<64>, !tla.vector<64xi32>, !tla.vector<64xi32>) -> !tla.vector<64xi32>
        %out_mask_or = "tla.where"(%mask_bitwise_or, %reg0, %zero_reg) : (!tla.mask<64>, !tla.vector<64xi32>, !tla.vector<64xi32>) -> !tla.vector<64xi32>
        %out_mask_xor = "tla.where"(%mask_bitwise_xor, %reg0, %zero_reg) : (!tla.mask<64>, !tla.vector<64xi32>, !tla.vector<64xi32>) -> !tla.vector<64xi32>
        %reg_bitwise_not = "tla.bitwise_not"(%reg0) : (!tla.vector<64xi32>) -> !tla.vector<64xi32>
        %reg_bitwise_and_all = "tla.bitwise_and"(%reg0, %reg1) : (!tla.vector<64xi32>, !tla.vector<64xi32>) -> !tla.vector<64xi32>
        %reg_bitwise_and = "tla.bitwise_and"(%reg_bitwise_and_all, %reg1, %all) : (!tla.vector<64xi32>, !tla.vector<64xi32>, !tla.mask<64>) -> !tla.vector<64xi32>
        %reg_bitwise_or_all = "tla.bitwise_or"(%reg0, %reg1) : (!tla.vector<64xi32>, !tla.vector<64xi32>) -> !tla.vector<64xi32>
        %reg_bitwise_or = "tla.bitwise_or"(%reg_bitwise_or_all, %reg1, %all) : (!tla.vector<64xi32>, !tla.vector<64xi32>, !tla.mask<64>) -> !tla.vector<64xi32>
        %reg_bitwise_xor_all = "tla.bitwise_xor"(%reg0, %reg1) : (!tla.vector<64xi32>, !tla.vector<64xi32>) -> !tla.vector<64xi32>
        %reg_bitwise_xor = "tla.bitwise_xor"(%reg_bitwise_xor_all, %reg1, %all) : (!tla.vector<64xi32>, !tla.vector<64xi32>, !tla.mask<64>) -> !tla.vector<64xi32>
        "tla.store"(%dst_mask_not_tile, %out_mask_not) <{operandSegmentSizes = array<i32: 1, 1, 0, 0, 0>}> : (!vec, !tla.vector<64xi32>) -> ()
        "tla.store"(%dst_mask_and_tile, %out_mask_and) <{operandSegmentSizes = array<i32: 1, 1, 0, 0, 0>}> : (!vec, !tla.vector<64xi32>) -> ()
        "tla.store"(%dst_mask_or_tile, %out_mask_or) <{operandSegmentSizes = array<i32: 1, 1, 0, 0, 0>}> : (!vec, !tla.vector<64xi32>) -> ()
        "tla.store"(%dst_mask_xor_tile, %out_mask_xor) <{operandSegmentSizes = array<i32: 1, 1, 0, 0, 0>}> : (!vec, !tla.vector<64xi32>) -> ()
        "tla.store"(%dst_reg_not_tile, %reg_bitwise_not) <{operandSegmentSizes = array<i32: 1, 1, 0, 0, 0>}> : (!vec, !tla.vector<64xi32>) -> ()
        "tla.store"(%dst_reg_and_tile, %reg_bitwise_and) <{operandSegmentSizes = array<i32: 1, 1, 0, 0, 0>}> : (!vec, !tla.vector<64xi32>) -> ()
        "tla.store"(%dst_reg_or_tile, %reg_bitwise_or) <{operandSegmentSizes = array<i32: 1, 1, 0, 0, 0>}> : (!vec, !tla.vector<64xi32>) -> ()
        "tla.store"(%dst_reg_xor_tile, %reg_bitwise_xor) <{operandSegmentSizes = array<i32: 1, 1, 0, 0, 0>}> : (!vec, !tla.vector<64xi32>) -> ()
      }) {mode = "simd"} : () -> ()
    }) : () -> ()
    return
  }
}
