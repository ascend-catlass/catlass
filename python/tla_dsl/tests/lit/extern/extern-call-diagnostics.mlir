// RUN: split-file %s %t
// RUN: not %tla_compile %t/opaque.mlir -o %t.out 2>&1 | %filecheck %s --check-prefix=MEMORY
// RUN: not %tla_compile %t/integer.mlir -o %t.out 2>&1 | %filecheck %s --check-prefix=MEMORY
// RUN: not %tla_compile %t/unsupported.mlir -o %t.out 2>&1 | %filecheck %s --check-prefix=OUTLINE
// RUN: not %tla_compile %t/pointee.mlir -o %t.out 2>&1 | %filecheck %s --check-prefix=SIGNATURE
// RUN: not %tla_compile %t/space.mlir -o %t.out 2>&1 | %filecheck %s --check-prefix=SIGNATURE
// RUN: not %tla_compile %t/scalar.mlir -o %t.out 2>&1 | %filecheck %s --check-prefix=SIGNATURE

// MEMORY: SIMD VF with an extern call requires a tensor memory access or an extern pointer argument
// OUTLINE: unsupported op while outlining tla.vec.func into vector helper: func.call
// OUTLINE: failed to outline enclosing SIMD VF for extern @consume
// SIGNATURE: external symbol @consume was called with incompatible function types

//--- opaque.mlir
module {
  tla.func @kernel() {
    "tla.vector"() ({
      "tla.vec.func"() ({
        tla.call_extern @opaque() : () -> ()
      }) : () -> ()
    }) : () -> ()
    tla.return
  }
}

//--- integer.mlir
module {
  tla.func @kernel(%n: i64) {
    "tla.vector"() ({
      "tla.vec.func"() ({
        tla.call_extern @opaque(%n) : (i64) -> ()
      }) : () -> ()
    }) : () -> ()
    tla.return
  }
}

//--- unsupported.mlir
module {
  func.func private @unknown(f32) -> f32
  tla.func @kernel(%x: f32) {
    %p = tla.alloc_ptr{size_bytes = 256} -> !tla.ptr<f32, ub, 256>
    "tla.vector"() ({
      "tla.vec.func"() ({
        %v = func.call @unknown(%x) : (f32) -> f32
        tla.call_extern @consume(%p, %v) : (!tla.ptr<f32, ub, 256>, f32) -> ()
      }) : () -> ()
    }) : () -> ()
    tla.return
  }
}

//--- pointee.mlir
module {
  func.func @kernel(%a: !tla.ptr<i32, ub, 32>, %b: !tla.ptr<f32, ub, 32>) {
    "tla.vector"() ({
      tla.call_extern @consume(%a) : (!tla.ptr<i32, ub, 32>) -> ()
      tla.call_extern @consume(%b) : (!tla.ptr<f32, ub, 32>) -> ()
    }) : () -> ()
    return
  }
}

//--- space.mlir
module {
  func.func @kernel(%a: !tla.ptr<f32, ub, 32>, %b: !tla.ptr<f32, gm, 32>) {
    "tla.vector"() ({
      tla.call_extern @consume(%a) : (!tla.ptr<f32, ub, 32>) -> ()
      tla.call_extern @consume(%b) : (!tla.ptr<f32, gm, 32>) -> ()
    }) : () -> ()
    return
  }
}

//--- scalar.mlir
module {
  func.func @kernel(%a: i64, %b: i32) {
    "tla.vector"() ({
      tla.call_extern @consume(%a) : (i64) -> ()
      tla.call_extern @consume(%b) : (i32) -> ()
    }) : () -> ()
    return
  }
}
