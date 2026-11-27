// RUN: %tla_compile %s -o - | %filecheck %s --implicit-check-not=tla.arch.store_bypass_dcache

module {
  tla.func @mixed_store(%ptr: !tla.ptr<i32, gm, 4>, %offset: index, %value: i32) {
    tla.cube {
      %dst = tla.ptr_add %ptr, %offset : !tla.ptr<i32, gm, 4>, index -> !tla.ptr<i32, gm, 4>
      %address = tla.ptrtoint %dst : !tla.ptr<i32, gm, 4> -> i64
      tla.arch.store_bypass_dcache %address, %value
    }
    "tla.vector"() ({
      %address = tla.ptrtoint %ptr : !tla.ptr<i32, gm, 4> -> i64
      tla.arch.store_bypass_dcache %address, %value
      tla.arch.store_bypass_dcache %address, %value
    }) : () -> ()
    tla.return
  }
}

// CHECK: func.func private @_mlir_ciface_tla_store_bypass_dcache_i32(i64, i32) attributes {{.*}}hivm.func_core_type = #hivm.func_core_type<AIC_OR_AIV>
// CHECK-LABEL: func.func @mixed_store_mix_aic(
// CHECK: %[[ADDR:.*]] = llvm.add
// CHECK: call @_mlir_ciface_tla_store_bypass_dcache_i32(%[[ADDR]], {{.*}}) : (i64, i32) -> ()
// CHECK-LABEL: func.func @mixed_store_mix_aiv(
// CHECK-COUNT-2: call @_mlir_ciface_tla_store_bypass_dcache_i32({{.*}}) : (i64, i32) -> ()
