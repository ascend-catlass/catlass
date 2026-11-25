// RUN: %tla_compile %s -o - | %filecheck %s

module {
  tla.func @bypass(%p8: !tla.ptr<i8, gm, 1>, %p16: !tla.ptr<ui16, gm, 2>,
                   %p32: !tla.ptr<i32, gm, 4>, %p64: !tla.ptr<ui64, gm, 8>, %offset: index) {
    tla.cube {
      %addr = tla.ptr_add %p8, %offset : !tla.ptr<i8, gm, 1>, index -> !tla.ptr<i8, gm, 1>
      %addr8 = tla.ptrtoint %addr : !tla.ptr<i8, gm, 1> -> i64
      %addr16 = tla.ptrtoint %p16 : !tla.ptr<ui16, gm, 2> -> i64
      %addr32 = tla.ptrtoint %p32 : !tla.ptr<i32, gm, 4> -> i64
      %addr64 = tla.ptrtoint %p64 : !tla.ptr<ui64, gm, 8> -> i64
      %a = tla.arch.load_bypass_dcache %addr8 : i8
      %b = tla.arch.load_bypass_dcache %addr8 : i8
      %c = tla.arch.load_bypass_dcache %addr16 : ui16
      %d = tla.arch.load_bypass_dcache %addr32 : i32
      %e = tla.arch.load_bypass_dcache %addr64 : ui64
    }
    tla.return
  }
}

// CHECK-LABEL: func.func @bypass(
// CHECK: %[[ADDR:.*]] = llvm.add
// CHECK: %[[PTR:.*]] = llvm.inttoptr %[[ADDR]] : i64 to !llvm.ptr<1>
// CHECK: hivm.hir.load_scalar %[[PTR]] : <1> -> i8
// CHECK: hivm.hir.load_scalar %[[PTR]] : <1> -> i8
// CHECK: hivm.hir.load_scalar {{.*}} : <1> -> i16
// CHECK: hivm.hir.load_scalar {{.*}} : <1> -> i32
// CHECK: hivm.hir.load_scalar {{.*}} : <1> -> i64
