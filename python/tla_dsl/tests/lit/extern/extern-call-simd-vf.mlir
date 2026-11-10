// RUN: %tla_compile %s -o - | %filecheck %s
// RUN: %tla_compile %s --mlir-print-ir-after=tla-lower-extern-call -o %t 2>&1 | %filecheck %s --check-prefix=EXTERN
// RUN: %tla_compile %s --mlir-print-ir-after=tla-vector-region-simd -o %t 2>&1 | %filecheck %s --check-prefix=SIMD

// EXTERN-LABEL: func.func @simd_extern(
// EXTERN: call @shared({{.*}}) : (!tla.ptr<f32, ub, 256>, i32) -> ()
// EXTERN: tla.vec.func
// EXTERN: tla.call_extern @shared

!fvec = !tla.tensor<!tla.layout<!tla.shape<64>, !tla.stride<1>, !tla.shape<64>, RowMajor>, !tla.coord<0>, !tla.ptr<f32, ub, 4>>

module {
  tla.func @simd_extern(%n: i32) {
    %ub = tla.alloc_ptr{size_bytes = 256} -> !tla.ptr<f32, ub, 256>
    "tla.vector"() ({
      tla.call_extern @shared(%ub, %n) : (!tla.ptr<f32, ub, 256>, i32) -> ()
      "tla.vec.func"() ({
        %zero = arith.constant 0 : i32
        %active = arith.cmpi sgt, %n, %zero : i32
        scf.if %active {
          tla.call_extern @shared(%ub, %n) : (!tla.ptr<f32, ub, 256>, i32) -> ()
        }
      }) {mode = "simd"} : () -> ()
    }) : () -> ()
    tla.return
  }

  func.func @with_store(%dst: memref<64xf32, #hivm.address_space<ub>>)
      attributes {hivm.func_core_type = #hivm.func_core_type<AIV>} {
    %zero = arith.constant 0 : index
    %one = arith.constant 1 : index
    %n = arith.constant 64 : index
    %tensor = tla.tensor_desc %dst shape [%one, %n, %one, %one] stride [%n, %one, %one, %one] origin_shape [%one, %n] coord [%zero, %zero] : memref<64xf32, #hivm.address_space<ub>> -> !fvec
    "tla.vec.func"() ({
      tla.call_extern @opaque() : () -> ()
      %shape = "tla.make_shape"() : () -> !tla.shape<64>
      %coord = "tla.make_coord"() : () -> !tla.coord<0>
      %tile = "tla.tile_view"(%tensor, %shape, %coord) : (!fvec, !tla.shape<64>, !tla.coord<0>) -> !fvec
      %f = arith.constant 1.0 : f32
      %v = "tla.full"(%f) : (f32) -> !tla.vector<64xf32>
      tla.store %tile, %v : !fvec, !tla.vector<64xf32>
    }) : () -> ()
    return
  }

  tla.func @addresses(%x: f32, %n: index) {
    %buf = tla.alloc_ptr{size_bytes = 256} -> !tla.ptr<f32, ub, 256>
    "tla.vector"() ({
      "tla.vec.func"() ({
        %zero = arith.constant 0 : index
        %one = arith.constant 1 : index
        scf.for %i = %zero to %n step %one {
          %p = tla.ptr_add %buf, %i : !tla.ptr<f32, ub, 256>, index -> !tla.ptr<f32, ub, 1>
          tla.call_extern @consume(%p, %x) : (!tla.ptr<f32, ub, 1>, f32) -> ()
        }
      }) : () -> ()
    }) : () -> ()
    tla.return
  }
}

// CHECK-DAG: func.func private @opaque() attributes {hivm.func_core_type = #hivm.func_core_type<AIV>}
// CHECK-DAG: func.func private @shared(i64, i32) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>}
// CHECK-LABEL: func.func @simd_extern(
// CHECK: call @shared
// CHECK: call @[[VF:[^(]+]]({{.*}}) {{.*}}hivm.vector_function
// CHECK: func.func private @[[VF]](
// CHECK-SAME: hivm.vector_function
// CHECK: call @shared({{.*}}) : (i64, i32) -> ()
// CHECK-LABEL: func.func @with_store(
// CHECK: call @[[STORE_VF:[^(]+]]({{.*}}) {{.*}}hivm.vector_function
// CHECK: func.func private @[[STORE_VF]](
// CHECK: call @opaque() : () -> ()
// CHECK: hivm.vstsx1.v64f32

// SIMD-LABEL: func.func @addresses(
// SIMD: call @[[VF:[^(]+]]({{.*}}) {{.*}}hivm.vector_function
// SIMD: func.func private @[[VF]](
// SIMD: scf.for %[[IV:[^ ]+]]
// SIMD: arith.index_cast %[[IV]]
// SIMD: arith.muli
// SIMD: %[[ADDR:.*]] = arith.addi
// SIMD: call @consume(%[[ADDR]], %{{[^)]+}}) : (i64, f32) -> ()
