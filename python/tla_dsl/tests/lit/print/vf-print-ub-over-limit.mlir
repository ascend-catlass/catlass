// RUN: not %tla_compile %s -o - 2>&1 | %filecheck %s
// CHECK: allocates 253960 bytes of UB, past the 253952-byte limit

module {
  tla.func @vf_print_ub_over_limit(%value: i32) {
    %fixed = tla.alloc_ptr{size_bytes = 253952} -> !tla.ptr<f32, ub, 256>
    "tla.vector"() ({
      "tla.vec.func"() ({
        tla.debug_print format "x={}"(%value) : (i32)
      }) {mode = "simd"} : () -> ()
    }) : () -> ()
    tla.return
  }
}
