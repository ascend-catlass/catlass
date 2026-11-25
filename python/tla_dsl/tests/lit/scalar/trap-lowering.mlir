// RUN: %tla_compile %s -o - | %filecheck %s --implicit-check-not=tla.trap

// A scalar-unit trap becomes a call to the common bitcode helper. Cube and
// vector place the same op on their own core, so the two calls come from the
// AIC and AIV functions respectively and share one AIC_OR_AIV declaration.
//
// The conditional case is the documented shape (`if divisor == 0: tla.trap()`):
// the trap sits inside an scf.if, so the call must stay in the branch instead of
// being hoisted or merged. A mixed kernel keeps the trap in both split halves.

!gm = !tla.tensor<!tla.layout<!tla.shape<16>, !tla.stride<1>, !tla.shape<16>, RowMajor>, !tla.coord<0>, !tla.ptr<i32, gm, 4>>

module {
  tla.func @trap_cube() {
    "tla.cube"() ({
      tla.trap
    }) : () -> ()
    tla.return
  }

  tla.func @trap_vector() {
    "tla.vector"() ({
      tla.trap
    }) : () -> ()
    tla.return
  }

  // The kernel body itself: no tla.cube / tla.vector wrapper.
  tla.func @trap_plain() {
    tla.trap
    tla.return
  }

  // The documented shape: the condition is decided in scalar code, the trap
  // stays in the branch inside the region.
  tla.func @trap_conditional(%src: !gm) {
    %c0 = arith.constant 0 : index
    %divisor = tla.scalar_load %src[%c0] : !gm -> i32
    %zero = arith.constant 0 : i32
    %is_zero = arith.cmpi eq, %divisor, %zero : i32
    "tla.vector"() ({
      scf.if %is_zero {
        tla.trap
      }
    }) : () -> ()
    tla.return
  }

  tla.func @trap_mixed() {
    "tla.cube"() ({
      tla.trap
    }) : () -> ()
    "tla.vector"() ({
      tla.trap
    }) : () -> ()
    tla.return
  }
}

// CHECK: func.func private @_mlir_ciface_tla_trap()
// CHECK-SAME: hacc.always_inline
// CHECK-SAME: hivm.func_core_type = #hivm.func_core_type<AIC_OR_AIV>
// CHECK-LABEL: func.func @trap_cube
// CHECK: call @_mlir_ciface_tla_trap() : () -> ()
// CHECK-LABEL: func.func @trap_vector
// CHECK: call @_mlir_ciface_tla_trap() : () -> ()
// CHECK-LABEL: func.func @trap_plain
// CHECK: call @_mlir_ciface_tla_trap() : () -> ()
// CHECK-LABEL: func.func @trap_conditional
// CHECK: cf.cond_br {{.*}}, ^[[TAKEN:bb[0-9]+]],
// CHECK: ^[[TAKEN]]:
// CHECK: call @_mlir_ciface_tla_trap() : () -> ()
// CHECK-LABEL: func.func @trap_mixed_mix_aic
// CHECK: call @_mlir_ciface_tla_trap() : () -> ()
// CHECK-LABEL: func.func @trap_mixed_mix_aiv
// CHECK: call @_mlir_ciface_tla_trap() : () -> ()
