// RUN: %tla_compile %s -o - --mlir-print-ir-after=tla-vector-region-simd 2>&1 | %filecheck %s

// A plain (non-i1) scalar tla.full lowers to the mask-capable AVE broadcast
// (all-true mask when none is given), which lowers to vdups.z.
// CHECK: ave.hir.broadcast

!t = !tla.tensor<!tla.layout<!tla.shape<64>, !tla.stride<1>, !tla.shape<64>, RowMajor>, !tla.coord<0>, !tla.ptr<f32, ub, 4>>

module {
  func.func @scalar_lane_broadcast(
      %src_memref: memref<64xf32, #hivm.address_space<ub>>,
      %dst_memref: memref<64xf32, #hivm.address_space<ub>>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c64 = arith.constant 64 : index
    %src = tla.tensor_desc %src_memref shape [%c1, %c64, %c1, %c1] stride [%c64, %c1, %c1, %c1] origin_shape [%c1, %c64] coord [%c0, %c0] : memref<64xf32, #hivm.address_space<ub>> -> !t
    %dst = tla.tensor_desc %dst_memref shape [%c1, %c64, %c1, %c1] stride [%c64, %c1, %c1, %c1] origin_shape [%c1, %c64] coord [%c0, %c0] : memref<64xf32, #hivm.address_space<ub>> -> !t
    "tla.vector"() ({
      "tla.vec.func"() ({
        %shape = "tla.make_shape"() : () -> !tla.shape<64>
        %coord = "tla.make_coord"() : () -> !tla.coord<0>
        %src_tile = "tla.tile_view"(%src, %shape, %coord) : (!t, !tla.shape<64>, !tla.coord<0>) -> !t
        %dst_tile = "tla.tile_view"(%dst, %shape, %coord) : (!t, !tla.shape<64>, !tla.coord<0>) -> !t
        %reg = "tla.load"(%src_tile) <{operandSegmentSizes = array<i32: 1, 0, 0>}> : (!t) -> !tla.vector<64xf32>
        %zero = arith.constant 0.000000e+00 : f32
        %full = "tla.full"(%zero) : (f32) -> !tla.vector<64xf32>
        %out = "tla.add"(%reg, %full) : (!tla.vector<64xf32>, !tla.vector<64xf32>) -> !tla.vector<64xf32>
        "tla.store"(%dst_tile, %out) <{operandSegmentSizes = array<i32: 1, 1, 0, 0>}> : (!t, !tla.vector<64xf32>) -> ()
      }) {mode = "simd"} : () -> ()
    }) : () -> ()
    return
  }

  // A func-argument scalar is captured into the vector helper as a trailing
  // block argument (collectVectorHelperScalarOperands) and broadcast from
  // there. The arith.constant source above only exercises the inline-clone
  // branch of lookupOrCloneScalarValue, so it never reaches the capture path.
  func.func @scalar_arg_broadcast(
      %src_memref: memref<64xf32, #hivm.address_space<ub>>,
      %dst_memref: memref<64xf32, #hivm.address_space<ub>>,
      %val: f32) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c64 = arith.constant 64 : index
    %src = tla.tensor_desc %src_memref shape [%c1, %c64, %c1, %c1] stride [%c64, %c1, %c1, %c1] origin_shape [%c1, %c64] coord [%c0, %c0] : memref<64xf32, #hivm.address_space<ub>> -> !t
    %dst = tla.tensor_desc %dst_memref shape [%c1, %c64, %c1, %c1] stride [%c64, %c1, %c1, %c1] origin_shape [%c1, %c64] coord [%c0, %c0] : memref<64xf32, #hivm.address_space<ub>> -> !t
    "tla.vector"() ({
      "tla.vec.func"() ({
        %shape = "tla.make_shape"() : () -> !tla.shape<64>
        %coord = "tla.make_coord"() : () -> !tla.coord<0>
        %src_tile = "tla.tile_view"(%src, %shape, %coord) : (!t, !tla.shape<64>, !tla.coord<0>) -> !t
        %dst_tile = "tla.tile_view"(%dst, %shape, %coord) : (!t, !tla.shape<64>, !tla.coord<0>) -> !t
        %reg = "tla.load"(%src_tile) <{operandSegmentSizes = array<i32: 1, 0, 0>}> : (!t) -> !tla.vector<64xf32>
        %full = "tla.full"(%val) : (f32) -> !tla.vector<64xf32>
        %out = "tla.add"(%reg, %full) : (!tla.vector<64xf32>, !tla.vector<64xf32>) -> !tla.vector<64xf32>
        "tla.store"(%dst_tile, %out) <{operandSegmentSizes = array<i32: 1, 1, 0, 0>}> : (!t, !tla.vector<64xf32>) -> ()
      }) {mode = "simd"} : () -> ()
    }) : () -> ()
    return
  }
}
// CHECK-LABEL: func.func @scalar_arg_broadcast(
// CHECK: ave.hir.broadcast %{{arg[0-9]+}},
