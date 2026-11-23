// RUN: %tla_compile %s -o - --mlir-print-ir-after=tla-vector-region-simd 2>&1 | %filecheck %s

// tla.pair_wise lowers straight to the Vcpadd regbase instruction. There is
// no dedicated AVE pair-wise op. The op only accepts kind="add". The
// destination keeps the full 256-byte container: pairwise sums occupy the low
// half and the upper half is 0. The required mask converts to a preg and is
// passed through unchanged.

!t_f32 = !tla.tensor<!tla.layout<!tla.shape<64>, !tla.stride<1>, !tla.shape<64>, RowMajor>, !tla.coord<0>, !tla.ptr<f32, ub, 4>>
!t_f16 = !tla.tensor<!tla.layout<!tla.shape<128>, !tla.stride<1>, !tla.shape<128>, RowMajor>, !tla.coord<0>, !tla.ptr<f16, ub, 2>>

module {
  // CHECK-LABEL: func.func @pair_wise_f32
  // CHECK: "hivm_regbaseintrins.intr.hivm.vcpadd.x"({{.*}}) : (vector<64xf32>, vector<256xi1>) -> vector<64xf32>
  func.func @pair_wise_f32(%src_memref: memref<64xf32, #hivm.address_space<ub>>, %dst_memref: memref<64xf32, #hivm.address_space<ub>>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c64 = arith.constant 64 : index
    %src = tla.tensor_desc %src_memref shape [%c1, %c64, %c1, %c1] stride [%c64, %c1, %c1, %c1] origin_shape [%c1, %c64] coord [%c0, %c0] : memref<64xf32, #hivm.address_space<ub>> -> !t_f32
    %dst = tla.tensor_desc %dst_memref shape [%c1, %c64, %c1, %c1] stride [%c64, %c1, %c1, %c1] origin_shape [%c1, %c64] coord [%c0, %c0] : memref<64xf32, #hivm.address_space<ub>> -> !t_f32
    "tla.vector"() ({
      "tla.vec.func"() ({
        %shape = "tla.make_shape"() : () -> !tla.shape<64>
        %coord = "tla.make_coord"() : () -> !tla.coord<0>
        %src_tile = "tla.tile_view"(%src, %shape, %coord) : (!t_f32, !tla.shape<64>, !tla.coord<0>) -> !t_f32
        %dst_tile = "tla.tile_view"(%dst, %shape, %coord) : (!t_f32, !tla.shape<64>, !tla.coord<0>) -> !t_f32
        %mask = "tla.create_mask"() {pattern = "ALL", dtype = f32} : () -> !tla.mask<64>
        %v = "tla.load"(%src_tile) : (!t_f32) -> !tla.vector<64xf32>
        %r = "tla.pair_wise"(%v, %mask) {kind = "add"} : (!tla.vector<64xf32>, !tla.mask<64>) -> !tla.vector<64xf32>
        "tla.store"(%dst_tile, %r) <{operandSegmentSizes = array<i32: 1, 1, 0, 0, 0>}> : (!t_f32, !tla.vector<64xf32>) -> ()
      }) {mode = "simd"} : () -> ()
    }) : () -> ()
    return
  }

  // CHECK-LABEL: func.func @pair_wise_f16
  // CHECK: "hivm_regbaseintrins.intr.hivm.vcpadd.x"({{.*}}) : (vector<128xf16>, vector<256xi1>) -> vector<128xf16>
  func.func @pair_wise_f16(%src_memref: memref<128xf16, #hivm.address_space<ub>>, %dst_memref: memref<128xf16, #hivm.address_space<ub>>) {
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
        %r = "tla.pair_wise"(%v, %mask) {kind = "add"} : (!tla.vector<128xf16>, !tla.mask<128>) -> !tla.vector<128xf16>
        "tla.store"(%dst_tile, %r) <{operandSegmentSizes = array<i32: 1, 1, 0, 0, 0>}> : (!t_f16, !tla.vector<128xf16>) -> ()
      }) {mode = "simd"} : () -> ()
    }) : () -> ()
    return
  }
}
