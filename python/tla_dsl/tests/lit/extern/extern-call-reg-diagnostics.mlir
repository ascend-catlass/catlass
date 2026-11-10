// RUN: split-file %s %t
// RUN: not %tla_compile %t/lanes.mlir -o %t.out 2>&1 | %filecheck %s --check-prefix=LANES
// RUN: not %tla_compile %t/dtype.mlir -o %t.out 2>&1 | %filecheck %s --check-prefix=DTYPE
// RUN: not %tla_compile %t/result.mlir -o %t.out 2>&1 | %filecheck %s --check-prefix=RESULT
// RUN: not %tla_compile %t/multi.mlir -o %t.out 2>&1 | %filecheck %s --check-prefix=MULTI
// RUN: not %tla_compile %t/signature.mlir -o %t.out 2>&1 | %filecheck %s --check-prefix=SIGNATURE
// RUN: not %tla_compile %t/memory.mlir -o %t.out 2>&1 | %filecheck %s --check-prefix=MEMORY
// RUN: not %tla_compile %t/cross.mlir -o %t.out 2>&1 | %filecheck %s --check-prefix=CROSS
// RUN: not %tla_compile %t/simt.mlir -o %t.out 2>&1 | %filecheck %s --check-prefix=SIMT
// RUN: not %tla_compile %t/outside.mlir -o %t.out 2>&1 | %filecheck %s --check-prefix=OUTSIDE
// RUN: not %tla_compile %t/signedness.mlir -o %t.out 2>&1 | %filecheck %s --check-prefix=SIGNEDNESS

// LANES: register arguments require a SIMD VF and a full 256-byte register
// DTYPE: register arguments require a SIMD VF and a full 256-byte register
// RESULT: register results require a SIMD VF and a full 256-byte register
// MULTI: supports at most one register result
// SIGNATURE: external symbol @make was called with incompatible function types
// MEMORY: SIMD VF with an extern call requires a tensor memory access or an extern pointer argument
// CROSS: register arguments must be defined in the same SIMD VF
// SIMT: not supported inside a SIMT tla.vec.func
// OUTSIDE: register results require a SIMD VF and a full 256-byte register
// SIGNEDNESS: external symbol @use was called with incompatible function types

//--- lanes.mlir
module {
  func.func @test(%addr: i64, %x: f32) {
    %p = tla.inttoptr %addr : i64 -> !tla.ptr<f32, ub, 4>
    "tla.vector"() ({
      "tla.vec.func"() ({
        %v = "tla.full"(%x) : (f32) -> !tla.vector<32xf32>
        tla.call_extern @use(%v, %p) : (!tla.vector<32xf32>, !tla.ptr<f32, ub, 4>) -> ()
      }) : () -> ()
    }) : () -> ()
    return
  }
}

//--- dtype.mlir
module {
  func.func @test(%addr: i64, %v: !tla.vector<32xf64>) {
    %p = tla.inttoptr %addr : i64 -> !tla.ptr<f32, ub, 4>
    "tla.vector"() ({
      "tla.vec.func"() ({
        tla.call_extern @use(%v, %p) : (!tla.vector<32xf64>, !tla.ptr<f32, ub, 4>) -> ()
      }) : () -> ()
    }) : () -> ()
    return
  }
}

//--- result.mlir
module {
  func.func @test(%addr: i64) {
    %p = tla.inttoptr %addr : i64 -> !tla.ptr<f32, ub, 4>
    "tla.vector"() ({
      "tla.vec.func"() ({
        %v = tla.call_extern @make(%p) : (!tla.ptr<f32, ub, 4>) -> f32
      }) : () -> ()
    }) : () -> ()
    return
  }
}

//--- multi.mlir
module {
  func.func @test(%addr: i64) {
    %p = tla.inttoptr %addr : i64 -> !tla.ptr<f32, ub, 4>
    "tla.vector"() ({
      "tla.vec.func"() ({
        %v:2 = tla.call_extern @make(%p) : (!tla.ptr<f32, ub, 4>) -> (!tla.vector<64xf32>, !tla.vector<64xf32>)
      }) : () -> ()
    }) : () -> ()
    return
  }
}

//--- signature.mlir
module {
  func.func @test(%addr: i64) {
    %p = tla.inttoptr %addr : i64 -> !tla.ptr<f32, ub, 4>
    "tla.vector"() ({
      "tla.vec.func"() ({
        %v = tla.call_extern @make(%p) : (!tla.ptr<f32, ub, 4>) -> !tla.vector<64xf32>
        tla.call_extern @make(%p) : (!tla.ptr<f32, ub, 4>) -> ()
      }) : () -> ()
    }) : () -> ()
    return
  }
}

//--- memory.mlir
module {
  func.func @test(%x: f32) {
    "tla.vector"() ({
      "tla.vec.func"() ({
        %v = "tla.full"(%x) : (f32) -> !tla.vector<64xf32>
        %r = tla.call_extern @pure(%v) : (!tla.vector<64xf32>) -> !tla.vector<64xf32>
      }) : () -> ()
    }) : () -> ()
    return
  }
}

//--- cross.mlir
module {
  func.func @test(%addr: i64, %external: !tla.vector<64xf32>) {
    %p = tla.inttoptr %addr : i64 -> !tla.ptr<f32, ub, 4>
    "tla.vector"() ({
      "tla.vec.func"() ({
        tla.call_extern @use(%external, %p) : (!tla.vector<64xf32>, !tla.ptr<f32, ub, 4>) -> ()
      }) : () -> ()
    }) : () -> ()
    return
  }
}

//--- simt.mlir
module {
  func.func @test(%addr: i64) {
    %p = tla.inttoptr %addr : i64 -> !tla.ptr<f32, ub, 4>
    "tla.vector"() ({
      "tla.vec.func"() ({
        %v = tla.call_extern @make(%p) : (!tla.ptr<f32, ub, 4>) -> !tla.vector<64xf32>
      }) {mode = "simt", thread_block_dim = array<i64: 64, 1, 1>} : () -> ()
    }) : () -> ()
    return
  }
}

//--- outside.mlir
module {
  func.func @test(%addr: i64) {
    %p = tla.inttoptr %addr : i64 -> !tla.ptr<f32, ub, 4>
    "tla.vector"() ({
      %v = tla.call_extern @make(%p) : (!tla.ptr<f32, ub, 4>) -> !tla.vector<64xf32>
    }) : () -> ()
    return
  }
}

//--- signedness.mlir
module {
  func.func @test(%addr: i64) {
    %p = tla.inttoptr %addr : i64 -> !tla.ptr<i8, ub, 4>
    "tla.vector"() ({
      "tla.vec.func"() ({
        %signed = tla.call_extern @read_signed(%p) : (!tla.ptr<i8, ub, 4>) -> !tla.vector<64xi32>
        %unsigned = tla.call_extern @read_unsigned(%p) : (!tla.ptr<i8, ub, 4>) -> !tla.vector<64xui32>
        tla.call_extern @use(%signed) : (!tla.vector<64xi32>) -> ()
        tla.call_extern @use(%unsigned) : (!tla.vector<64xui32>) -> ()
      }) : () -> ()
    }) : () -> ()
    return
  }
}
