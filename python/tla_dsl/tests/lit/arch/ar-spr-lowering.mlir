// RUN: %tla_compile %s -o - > %t 2>&1
// RUN: %filecheck %s --check-prefix=DECL < %t
// RUN: %filecheck %s --check-prefix=VF < %t
// RUN: %filecheck %s --check-prefix=SCALAR < %t
// RUN: %filecheck %s --check-prefix=ABSENT < %t

// Each AR operation becomes a call to a meta-op bitcode helper. The clear
// picks its helper from where it executes: the in-VF one when it shares a vector
// function with the stores, the scalar one when it stands on its own.

!t = !tla.tensor<!tla.layout<!tla.shape<64>, !tla.stride<1>, !tla.shape<64>, RowMajor>, !tla.coord<0>, !tla.ptr<f32, ub, 4>>

module {
  func.func @ar_in_vec_func(
      %src_memref: memref<64xf32, #hivm.address_space<ub>>,
      %dst_memref: memref<64xf32, #hivm.address_space<ub>>,
      %out: memref<1xi64>) {
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
        tla.arch.clear_ar
        %reg = "tla.load"(%src_tile) <{operandSegmentSizes = array<i32: 1, 0, 0>}> : (!t) -> !tla.vector<64xf32>
        "tla.store"(%dst_tile, %reg) <{operandSegmentSizes = array<i32: 1, 1, 0, 0, 0>}> : (!t, !tla.vector<64xf32>) -> ()
      }) {mode = "simd"} : () -> ()
      %count = tla.arch.get_ar -> i64
      memref.store %count, %out[%c0] : memref<1xi64>
    }) : () -> ()
    return
  }

  // A vec.func holding nothing but the clear has no tile work to outline, so it
  // is flattened into the enclosing tla.vector() and the clear takes the scalar
  // helper -- the one that brings its own vector function. Two clears and two
  // reads here: neither pair may collapse, or the second read would report the
  // count from before the second clear.
  func.func @ar_scalar_only(%out: memref<2xi64>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    "tla.vector"() ({
      "tla.vec.func"() ({
        tla.arch.clear_ar
      }) {mode = "simd"} : () -> ()
      %first = tla.arch.get_ar -> i64
      memref.store %first, %out[%c0] : memref<2xi64>
      tla.arch.clear_ar
      %second = tla.arch.get_ar -> i64
      memref.store %second, %out[%c1] : memref<2xi64>
    }) : () -> ()
    return
  }
}

// All three helpers are private declarations the linker resolves and all three
// are AIV: AR is a vector-unit register and the helpers only exist in the
// vector meta-op bitcode, so an AIC placement is a link error rather than a
// silently wrong read. The two scalar-unit entries are always_inline, so each
// costs its one instruction rather than a call.
// DECL-DAG: func.func private @_mlir_ciface_tla_get_ar() -> i64 attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>}
// DECL-DAG: func.func private @_mlir_ciface_tla_clear_ar() attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>}
// DECL-DAG: func.func private @squeeze_stream_clear() attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}

// The in-VF clear lands inside the outlined vector function, beside the load
// and store it leads, and not in the scalar body that calls that function. The
// read stays in the scalar body.
// VF-LABEL: func.func @ar_in_vec_func
// VF: call @vector_region_0
// VF-NOT: _mlir_ciface_tla_clear_ar
// VF-NOT: squeeze_stream_clear
// VF: call @_mlir_ciface_tla_get_ar() : () -> i64
// VF-LABEL: func.func private @vector_region_0
// VF-SAME: hivm.vector_function
// VF: call @squeeze_stream_clear() : () -> ()
// VF: hivm_regbaseintrins.intr.hivm.vldsx1
// VF: hivm_regbaseintrins.intr.hivm.vstsx1

// The flattened clear takes the scalar helper, and both pairs survive: two
// clears and two reads, in source order.
// SCALAR-LABEL: func.func @ar_scalar_only
// SCALAR: call @_mlir_ciface_tla_clear_ar() : () -> ()
// SCALAR: call @_mlir_ciface_tla_get_ar() : () -> i64
// SCALAR: call @_mlir_ciface_tla_clear_ar() : () -> ()
// SCALAR: call @_mlir_ciface_tla_get_ar() : () -> i64

// Nothing of the TLA ops survives, and the clear-only region never reaches the
// in-VF helper: there is no vector function left for it to sit in.
// ABSENT-NOT: tla.arch.get_ar
// ABSENT-NOT: tla.arch.clear_ar
