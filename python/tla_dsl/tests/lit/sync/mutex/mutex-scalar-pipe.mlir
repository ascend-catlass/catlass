// RUN: %tla_compile %s -o - | %filecheck %s

// PIPE_S is the one pipe both cores accept for get_buf / rls_buf, so bc defines
// the stub pair in both Cube/mutex.cpp and Vector/mutex.cpp. This module reaches
// the same declaration from a cube and a vector function, so the pass stamps it
// AIC_OR_AIV below.

module {

  tla.func @mutex_scalar_pipe_cube() {
    "tla.cube"() ({
      %mutex = tla.mutex "l1_scalar" {id = 2 : i64} -> !tla.mutex
      tla.mutex_lock %mutex [#tla.pipe<scalar>] : !tla.mutex
      tla.mutex_unlock %mutex [#tla.pipe<scalar>] : !tla.mutex
    }) : () -> ()
    tla.return
  }

  tla.func @mutex_scalar_pipe_vector() {
    "tla.vector"() ({
      %mutex = tla.mutex "ub_scalar" {id = 5 : i64} -> !tla.mutex
      tla.mutex_lock %mutex [#tla.pipe<scalar>] : !tla.mutex
      tla.mutex_unlock %mutex [#tla.pipe<scalar>] : !tla.mutex
    }) : () -> ()
    tla.return
  }
}

// Both functions reach the same declaration, so the pass stamps it AIC_OR_AIV.
// CHECK-DAG: func.func private @get_buf_s(i8) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIC_OR_AIV>}
// CHECK-DAG: func.func private @rls_buf_s(i8) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIC_OR_AIV>}
// CHECK-LABEL: func.func @mutex_scalar_pipe_cube()
// CHECK-DAG: [[CUBE_ID:%.*]] = llvm.mlir.constant(2 : i8) : i8
// CHECK: call @get_buf_s([[CUBE_ID]])
// CHECK: call @rls_buf_s([[CUBE_ID]])
// CHECK-LABEL: func.func @mutex_scalar_pipe_vector()
// CHECK-DAG: [[VEC_ID:%.*]] = llvm.mlir.constant(5 : i8) : i8
// CHECK: call @get_buf_s([[VEC_ID]])
// CHECK: call @rls_buf_s([[VEC_ID]])
// CHECK-NOT: tla.mutex
