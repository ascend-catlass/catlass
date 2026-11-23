// RUN: %tla_compile %s -o - > %t 2>&1
// RUN: %filecheck %s --check-prefix=CALLS < %t
// RUN: %filecheck %s --check-prefix=ABSENT < %t

module {
  func.func @dcache_maintenance(%address: i64) {
    "tla.vector"() ({
      %ptr = tla.inttoptr %address : i64 -> !tla.ptr<i32, gm, 4>
      %byte_address = tla.ptrtoint %ptr : !tla.ptr<i32, gm, 4> -> i64
      tla.arch.dcci_cacheline %byte_address
      %three = arith.constant 3 : index
      %offset_ptr = tla.ptr_add %ptr, %three : !tla.ptr<i32, gm, 4>, index -> !tla.ptr<i32, gm, 4>
      %offset_address = tla.ptrtoint %offset_ptr : !tla.ptr<i32, gm, 4> -> i64
      tla.arch.dcci_cacheline %offset_address
      tla.arch.dcci_cachelines <all>
      tla.arch.dcci_cachelines <gm_l2>
      tla.arch.dcci_cachelines <st_atomic>
      tla.arch.dci
    }) : () -> ()
    func.return
  }
}

// DCI lowers to an inline bitcode helper.
// CALLS-DAG: func.func private @_mlir_ciface_tla_dci() attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIC_OR_AIV>}

// The operations stay in source order -- each one is a separate cache
// operation -- and the addressed form carries the byte address as the
// addrspace(1) pointer the intrin op takes.
// CALLS: func.func @dcache_maintenance
// CALLS-SAME: %[[ADDRESS:[A-Za-z0-9_]+]]: i64
// CALLS: %[[OFFSET_BYTES:.*]] = llvm.mlir.constant(12 : i64) : i64
// CALLS: %[[PTR:.*]] = llvm.inttoptr %[[ADDRESS]] : i64 to !llvm.ptr<1>
// CALLS: "hivm.intr.hivm.DCCI.DST"(%[[PTR]], %{{.*}}, %{{.*}}) : (!llvm.ptr<1>, i64, i64) -> ()
// CALLS: %[[OFFSET_ADDR:.*]] = llvm.add %[[ADDRESS]], %[[OFFSET_BYTES]] : i64
// CALLS: %[[OFFSET_PTR:.*]] = llvm.inttoptr %[[OFFSET_ADDR]] : i64 to !llvm.ptr<1>
// CALLS: "hivm.intr.hivm.DCCI.DST"(%[[OFFSET_PTR]], %{{.*}}, %{{.*}}) : (!llvm.ptr<1>, i64, i64) -> ()
// CALLS: hivm.hir.dcci(<all_cache_lines>, <all>)
// CALLS: hivm.hir.dcci(<all_cache_lines>, <out>)
// CALLS: hivm.hir.dcci(<all_cache_lines>, <atomic>)
// CALLS: call @_mlir_ciface_tla_dci() : () -> ()

// The TLA ops are fully lowered. Both DCCI forms use HIVM.
// ABSENT-NOT: tla.arch.dcci_cacheline
// ABSENT-NOT: tla.arch.dci
// ABSENT-NOT: tla.inttoptr
// ABSENT-NOT: tla.ptrtoint
// ABSENT-NOT: _mlir_ciface_tla_dcci
