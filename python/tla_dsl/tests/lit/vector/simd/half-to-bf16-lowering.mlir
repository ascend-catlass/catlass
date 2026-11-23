// RUN: %tla_compile %s -o - --mlir-print-ir-after=tla-vector-region-simd 2>&1 | %filecheck %s --implicit-check-not=ave.hir.vextf --implicit-check-not=ave.hir.vtruncf --implicit-check-not='vector<64xf32>'

// The direct conversion preserves all 128 lanes and consumes rounding and the
// B16 predicate without an even/odd selector or an f32 intermediate.
!src = !tla.tensor<!tla.layout<!tla.shape<128>, !tla.stride<1>, !tla.shape<128>, RowMajor>, !tla.coord<0>, !tla.ptr<f16, ub, 2>>
!dst = !tla.tensor<!tla.layout<!tla.shape<128>, !tla.stride<1>, !tla.shape<128>, RowMajor>, !tla.coord<0>, !tla.ptr<bf16, ub, 2>>

module {
// CHECK-LABEL: func.func @half_to_bf16_round
// CHECK: %[[MASK:.*]] = ave.hir.pge <ALL> {functionType = #ave.func_dist_type<pb16>} : vector<256xi1>
// CHECK: %[[RND:.*]] = arith.constant 1 : i32
// CHECK: "hivm_regbaseintrins.intr.hivm.vcvtff.f162bf16.x"(%{{.*}}, %[[MASK]], %[[RND]]) : (vector<128xf16>, vector<256xi1>, i32) -> vector<128xbf16>
  func.func @half_to_bf16_round(%sm: memref<128xf16, #hivm.address_space<ub>>,
      %dm: memref<128xbf16, #hivm.address_space<ub>>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %cn = arith.constant 128 : index
    %s = tla.tensor_desc %sm shape [%c1, %cn, %c1, %c1] stride [%cn, %c1, %c1, %c1] origin_shape [%c1, %cn] coord [%c0, %c0] : memref<128xf16, #hivm.address_space<ub>> -> !src
    %d = tla.tensor_desc %dm shape [%c1, %cn, %c1, %c1] stride [%cn, %c1, %c1, %c1] origin_shape [%c1, %cn] coord [%c0, %c0] : memref<128xbf16, #hivm.address_space<ub>> -> !dst
    "tla.vector"() ({
      "tla.vec.func"() ({
        %shape = "tla.make_shape"() : () -> !tla.shape<128>
        %coord = "tla.make_coord"() : () -> !tla.coord<0>
        %st = "tla.tile_view"(%s, %shape, %coord) : (!src, !tla.shape<128>, !tla.coord<0>) -> !src
        %dt = "tla.tile_view"(%d, %shape, %coord) : (!dst, !tla.shape<128>, !tla.coord<0>) -> !dst
        %r = "tla.load"(%st) : (!src) -> !tla.vector<128xf16>
        %o = "tla.cast"(%r) <{trait = array<i32: 0, 0, 0>}> : (!tla.vector<128xf16>) -> !tla.vector<128xbf16>
        "tla.store"(%dt, %o) <{operandSegmentSizes = array<i32: 1, 1, 0, 0, 0>}> : (!dst, !tla.vector<128xbf16>) -> ()
      }) {mode = "simd"} : () -> ()
    }) : () -> ()
    return
  }

// CHECK-LABEL: func.func @half_to_bf16_floor
// CHECK: %[[MASK:.*]] = ave.hir.pge <M4> {functionType = #ave.func_dist_type<pb16>} : vector<256xi1>
// CHECK: %[[RND:.*]] = arith.constant 2 : i32
// CHECK: "hivm_regbaseintrins.intr.hivm.vcvtff.f162bf16.x"(%{{.*}}, %[[MASK]], %[[RND]]) : (vector<128xf16>, vector<256xi1>, i32) -> vector<128xbf16>
  func.func @half_to_bf16_floor(%sm: memref<128xf16, #hivm.address_space<ub>>,
      %dm: memref<128xbf16, #hivm.address_space<ub>>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %cn = arith.constant 128 : index
    %s = tla.tensor_desc %sm shape [%c1, %cn, %c1, %c1] stride [%cn, %c1, %c1, %c1] origin_shape [%c1, %cn] coord [%c0, %c0] : memref<128xf16, #hivm.address_space<ub>> -> !src
    %d = tla.tensor_desc %dm shape [%c1, %cn, %c1, %c1] stride [%cn, %c1, %c1, %c1] origin_shape [%c1, %cn] coord [%c0, %c0] : memref<128xbf16, #hivm.address_space<ub>> -> !dst
    "tla.vector"() ({
      "tla.vec.func"() ({
        %shape = "tla.make_shape"() : () -> !tla.shape<128>
        %coord = "tla.make_coord"() : () -> !tla.coord<0>
        %st = "tla.tile_view"(%s, %shape, %coord) : (!src, !tla.shape<128>, !tla.coord<0>) -> !src
        %dt = "tla.tile_view"(%d, %shape, %coord) : (!dst, !tla.shape<128>, !tla.coord<0>) -> !dst
        %r = "tla.load"(%st) : (!src) -> !tla.vector<128xf16>
        %mask = "tla.create_mask"() {pattern = "M4", dtype = f16} : () -> !tla.mask<128>
        %o = "tla.cast"(%r, %mask) <{trait = array<i32: 1, 0, 1>}> : (!tla.vector<128xf16>, !tla.mask<128>) -> !tla.vector<128xbf16>
        "tla.store"(%dt, %o) <{operandSegmentSizes = array<i32: 1, 1, 0, 0, 0>}> : (!dst, !tla.vector<128xbf16>) -> ()
      }) {mode = "simd"} : () -> ()
    }) : () -> ()
    return
  }

// CHECK-LABEL: func.func @half_to_bf16_ceil
// CHECK: %[[MASK:.*]] = ave.hir.pge <ALL> {functionType = #ave.func_dist_type<pb16>} : vector<256xi1>
// CHECK: %[[RND:.*]] = arith.constant 3 : i32
// CHECK: "hivm_regbaseintrins.intr.hivm.vcvtff.f162bf16.x"(%{{.*}}, %[[MASK]], %[[RND]]) : (vector<128xf16>, vector<256xi1>, i32) -> vector<128xbf16>
  func.func @half_to_bf16_ceil(%sm: memref<128xf16, #hivm.address_space<ub>>,
      %dm: memref<128xbf16, #hivm.address_space<ub>>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %cn = arith.constant 128 : index
    %s = tla.tensor_desc %sm shape [%c1, %cn, %c1, %c1] stride [%cn, %c1, %c1, %c1] origin_shape [%c1, %cn] coord [%c0, %c0] : memref<128xf16, #hivm.address_space<ub>> -> !src
    %d = tla.tensor_desc %dm shape [%c1, %cn, %c1, %c1] stride [%cn, %c1, %c1, %c1] origin_shape [%c1, %cn] coord [%c0, %c0] : memref<128xbf16, #hivm.address_space<ub>> -> !dst
    "tla.vector"() ({
      "tla.vec.func"() ({
        %shape = "tla.make_shape"() : () -> !tla.shape<128>
        %coord = "tla.make_coord"() : () -> !tla.coord<0>
        %st = "tla.tile_view"(%s, %shape, %coord) : (!src, !tla.shape<128>, !tla.coord<0>) -> !src
        %dt = "tla.tile_view"(%d, %shape, %coord) : (!dst, !tla.shape<128>, !tla.coord<0>) -> !dst
        %r = "tla.load"(%st) : (!src) -> !tla.vector<128xf16>
        %o = "tla.cast"(%r) <{trait = array<i32: 0, 0, 2>}> : (!tla.vector<128xf16>) -> !tla.vector<128xbf16>
        "tla.store"(%dt, %o) <{operandSegmentSizes = array<i32: 1, 1, 0, 0, 0>}> : (!dst, !tla.vector<128xbf16>) -> ()
      }) {mode = "simd"} : () -> ()
    }) : () -> ()
    return
  }

// CHECK-LABEL: func.func @half_to_bf16_trunc
// CHECK: %[[MASK:.*]] = ave.hir.pge <M4> {functionType = #ave.func_dist_type<pb16>} : vector<256xi1>
// CHECK: %[[RND:.*]] = arith.constant 4 : i32
// CHECK: "hivm_regbaseintrins.intr.hivm.vcvtff.f162bf16.x"(%{{.*}}, %[[MASK]], %[[RND]]) : (vector<128xf16>, vector<256xi1>, i32) -> vector<128xbf16>
  func.func @half_to_bf16_trunc(%sm: memref<128xf16, #hivm.address_space<ub>>,
      %dm: memref<128xbf16, #hivm.address_space<ub>>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %cn = arith.constant 128 : index
    %s = tla.tensor_desc %sm shape [%c1, %cn, %c1, %c1] stride [%cn, %c1, %c1, %c1] origin_shape [%c1, %cn] coord [%c0, %c0] : memref<128xf16, #hivm.address_space<ub>> -> !src
    %d = tla.tensor_desc %dm shape [%c1, %cn, %c1, %c1] stride [%cn, %c1, %c1, %c1] origin_shape [%c1, %cn] coord [%c0, %c0] : memref<128xbf16, #hivm.address_space<ub>> -> !dst
    "tla.vector"() ({
      "tla.vec.func"() ({
        %shape = "tla.make_shape"() : () -> !tla.shape<128>
        %coord = "tla.make_coord"() : () -> !tla.coord<0>
        %st = "tla.tile_view"(%s, %shape, %coord) : (!src, !tla.shape<128>, !tla.coord<0>) -> !src
        %dt = "tla.tile_view"(%d, %shape, %coord) : (!dst, !tla.shape<128>, !tla.coord<0>) -> !dst
        %r = "tla.load"(%st) : (!src) -> !tla.vector<128xf16>
        %mask = "tla.create_mask"() {pattern = "M4", dtype = f16} : () -> !tla.mask<128>
        %o = "tla.cast"(%r, %mask) <{trait = array<i32: 1, 0, 3>}> : (!tla.vector<128xf16>, !tla.mask<128>) -> !tla.vector<128xbf16>
        "tla.store"(%dt, %o) <{operandSegmentSizes = array<i32: 1, 1, 0, 0, 0>}> : (!dst, !tla.vector<128xbf16>) -> ()
      }) {mode = "simd"} : () -> ()
    }) : () -> ()
    return
  }

}
