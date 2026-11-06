// RUN: %tla_compile %s -o - --mlir-print-ir-after=tla-vector-region-simd 2>&1 | %filecheck %s

// tla.reduce_intra_block lowers straight to the Vcg group-reduction regbase
// instructions. The container value is already a full 256-byte register
// vector, so there is no pad; the user mask converts to a preg and passes
// through unchanged. ADD on 16-bit sources widens to a 32-bit destination
// (same as vcadd), which is returned as-is with no Vcvtii cast back.

!t_i32 = !tla.tensor<!tla.layout<!tla.shape<64>, !tla.stride<1>, !tla.shape<64>, RowMajor>, !tla.coord<0>, !tla.ptr<i32, ub, 4>>
!t_ui16 = !tla.tensor<!tla.layout<!tla.shape<128>, !tla.stride<1>, !tla.shape<128>, RowMajor>, !tla.coord<0>, !tla.ptr<ui16, ub, 2>>
!t_f16 = !tla.tensor<!tla.layout<!tla.shape<128>, !tla.stride<1>, !tla.shape<128>, RowMajor>, !tla.coord<0>, !tla.ptr<f16, ub, 2>>
!t_dst_i32 = !tla.tensor<!tla.layout<!tla.shape<64>, !tla.stride<1>, !tla.shape<64>, RowMajor>, !tla.coord<0>, !tla.ptr<i32, ub, 4>>

module {
  // CHECK-LABEL: func.func @rdb_i32_add
  // i32 ADD is single-step on the signless container; the mask preg is a full
  // 256-byte register vector.
  // CHECK: "hivm_regbaseintrins.intr.hivm.vcgadd.s.x"({{.*}}) : (vector<64xi32>, vector<256xi1>) -> vector<64xi32>
  // CHECK-NOT: "hivm_regbaseintrins.intr.hivm.vcgadd.u.x"
  func.func @rdb_i32_add(%src_memref: memref<64xi32, #hivm.address_space<ub>>, %dst_memref: memref<64xi32, #hivm.address_space<ub>>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c64 = arith.constant 64 : index
    %src = tla.tensor_desc %src_memref shape [%c1, %c64, %c1, %c1] stride [%c64, %c1, %c1, %c1] origin_shape [%c1, %c64] coord [%c0, %c0] : memref<64xi32, #hivm.address_space<ub>> -> !t_i32
    %dst = tla.tensor_desc %dst_memref shape [%c1, %c64, %c1, %c1] stride [%c64, %c1, %c1, %c1] origin_shape [%c1, %c64] coord [%c0, %c0] : memref<64xi32, #hivm.address_space<ub>> -> !t_i32
    "tla.vector"() ({
      "tla.vec.func"() ({
        %shape = "tla.make_shape"() : () -> !tla.shape<64>
        %coord = "tla.make_coord"() : () -> !tla.coord<0>
        %src_tile = "tla.tile_view"(%src, %shape, %coord) : (!t_i32, !tla.shape<64>, !tla.coord<0>) -> !t_i32
        %dst_tile = "tla.tile_view"(%dst, %shape, %coord) : (!t_i32, !tla.shape<64>, !tla.coord<0>) -> !t_i32
        %mask = "tla.create_mask"() {pattern = "ALL", dtype = i32} : () -> !tla.mask<64>
        %v = "tla.load"(%src_tile) : (!t_i32) -> !tla.vector<64xi32>
        %r = "tla.reduce_intra_block"(%v, %mask) {kind = "add"} : (!tla.vector<64xi32>, !tla.mask<64>) -> !tla.vector<64xi32>
        "tla.store"(%dst_tile, %r) <{operandSegmentSizes = array<i32: 1, 1, 0, 0>}> : (!t_i32, !tla.vector<64xi32>) -> ()
      }) {mode = "simd"} : () -> ()
    }) : () -> ()
    return
  }

  // CHECK-LABEL: func.func @rdb_u16_add
  // u16 ADD widens: the instruction dst is 32-bit, but the LLVM intrinsic
  // signature needs a same-type source, so the Vcgadd runs on the 16-bit
  // container and the result register is re-presented as 64 x i32 by a
  // zero-cost bitcast. No Vcvtii cast back.
  // CHECK: "hivm_regbaseintrins.intr.hivm.vcgadd.u.x"({{.*}}) : (vector<128xi16>, vector<256xi1>) -> vector<128xi16>
  // CHECK: vector.bitcast %{{.*}} : vector<128xi16> to vector<64xi32>
  // CHECK-NOT: "hivm_regbaseintrins.intr.hivm.vcvtii
  func.func @rdb_u16_add(%src_memref: memref<128xui16, #hivm.address_space<ub>>, %dst_memref: memref<64xi32, #hivm.address_space<ub>>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c128 = arith.constant 128 : index
    %c64 = arith.constant 64 : index
    %src = tla.tensor_desc %src_memref shape [%c1, %c128, %c1, %c1] stride [%c128, %c1, %c1, %c1] origin_shape [%c1, %c128] coord [%c0, %c0] : memref<128xui16, #hivm.address_space<ub>> -> !t_ui16
    %dst = tla.tensor_desc %dst_memref shape [%c1, %c64, %c1, %c1] stride [%c64, %c1, %c1, %c1] origin_shape [%c1, %c64] coord [%c0, %c0] : memref<64xi32, #hivm.address_space<ub>> -> !t_dst_i32
    "tla.vector"() ({
      "tla.vec.func"() ({
        %shape = "tla.make_shape"() : () -> !tla.shape<128>
        %dst_shape = "tla.make_shape"() : () -> !tla.shape<64>
        %coord = "tla.make_coord"() : () -> !tla.coord<0>
        %src_tile = "tla.tile_view"(%src, %shape, %coord) : (!t_ui16, !tla.shape<128>, !tla.coord<0>) -> !t_ui16
        %dst_tile = "tla.tile_view"(%dst, %dst_shape, %coord) : (!t_dst_i32, !tla.shape<64>, !tla.coord<0>) -> !t_dst_i32
        %mask = "tla.create_mask"() {pattern = "ALL", dtype = ui16} : () -> !tla.mask<128>
        %v = "tla.load"(%src_tile) : (!t_ui16) -> !tla.vector<128xui16>
        %r = "tla.reduce_intra_block"(%v, %mask) {kind = "add"} : (!tla.vector<128xui16>, !tla.mask<128>) -> !tla.vector<64xui32>
        "tla.store"(%dst_tile, %r) <{operandSegmentSizes = array<i32: 1, 1, 0, 0>}> : (!t_dst_i32, !tla.vector<64xui32>) -> ()
      }) {mode = "simd"} : () -> ()
    }) : () -> ()
    return
  }

  // CHECK-LABEL: func.func @rdb_f16_max_min
  // Float MAX/MIN go through the S variants.
  // CHECK: "hivm_regbaseintrins.intr.hivm.vcgmax.s.x"({{.*}}) : (vector<128xf16>, vector<256xi1>) -> vector<128xf16>
  // CHECK: "hivm_regbaseintrins.intr.hivm.vcgmin.s.x"({{.*}}) : (vector<128xf16>, vector<256xi1>) -> vector<128xf16>
  func.func @rdb_f16_max_min(%src_memref: memref<128xf16, #hivm.address_space<ub>>, %dst_memref: memref<128xf16, #hivm.address_space<ub>>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c128 = arith.constant 128 : index
    %src = tla.tensor_desc %src_memref shape [%c1, %c128, %c1, %c1] stride [%c128, %c1, %c1, %c1] origin_shape [%c1, %c128] coord [%c0, %c0] : memref<128xf16, #hivm.address_space<ub>> -> !t_f16
    %dst = tla.tensor_desc %dst_memref shape [%c1, %c128, %c1, %c1] stride [%c128, %c1, %c1, %c1] origin_shape [%c1, %c128] coord [%c0, %c0] : memref<128xf16, #hivm.address_space<ub>> -> !t_f16
    "tla.vector"() ({
      "tla.vec.func"() ({
        %shape = "tla.make_shape"() : () -> !tla.shape<128>
        %coord = "tla.make_coord"() : () -> !tla.coord<0>
        %src_tile = "tla.tile_view"(%src, %shape, %coord) : (!t_f16, !tla.shape<128>, !tla.coord<0>) -> !t_f16
        %dst_tile = "tla.tile_view"(%dst, %shape, %coord) : (!t_f16, !tla.shape<128>, !tla.coord<0>) -> !t_f16
        %mask = "tla.create_mask"() {pattern = "ALL", dtype = f16} : () -> !tla.mask<128>
        %v = "tla.load"(%src_tile) : (!t_f16) -> !tla.vector<128xf16>
        %max_r = "tla.reduce_intra_block"(%v, %mask) {kind = "max"} : (!tla.vector<128xf16>, !tla.mask<128>) -> !tla.vector<128xf16>
        %min_r = "tla.reduce_intra_block"(%v, %mask) {kind = "min"} : (!tla.vector<128xf16>, !tla.mask<128>) -> !tla.vector<128xf16>
        "tla.store"(%dst_tile, %max_r) <{operandSegmentSizes = array<i32: 1, 1, 0, 0>}> : (!t_f16, !tla.vector<128xf16>) -> ()
        "tla.store"(%dst_tile, %min_r) <{operandSegmentSizes = array<i32: 1, 1, 0, 0>}> : (!t_f16, !tla.vector<128xf16>) -> ()
      }) {mode = "simd"} : () -> ()
    }) : () -> ()
    return
  }
}
