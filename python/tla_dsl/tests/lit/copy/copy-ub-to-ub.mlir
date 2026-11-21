// RUN: %tla_compile %s -o - | %filecheck %s
// RUN: sed -e 's/f32/i8/g' -e 's/copy_ub_to_ub/copy_ub_to_ub_i8/' %s | %tla_compile - -o - | %filecheck %s --check-prefix=CHECK-I8
// RUN: sed -e 's/f32/i1/g' -e 's/copy_ub_to_ub/copy_ub_to_ub_i1/' %s | %tla_compile - -o - | %filecheck %s --check-prefix=CHECK-BOOL

// tla.copy with UB on both sides. It is served by DataCopy, which lowers to the
// dedicated MOV.UB.TO.UB move rather than a vector-register round trip.
// Checked at the IR level: the route must reach the runtime call rather than
// being rejected as an unsupported copy route.
// CHECK: copy_ub_RowMajor_to_ub_RowMajor_float
//
// DataCopy moves 32-byte blocks, so 1-byte element types take the same route --
// the width only has to describe the block arithmetic. 64 i8 elements is two
// whole blocks, so this is a legal row.
// CHECK-I8: copy_ub_RowMajor_to_ub_RowMajor_int8_t
//
// Bool / i1 is the same 1-byte staging width (AscendC b8); symbol suffix is bool.
// CHECK-BOOL: copy_ub_RowMajor_to_ub_RowMajor_bool

!t = !tla.tensor<!tla.layout<!tla.shape<64>, !tla.stride<1>, !tla.shape<64>, RowMajor>, !tla.coord<0>, !tla.ptr<f32, ub, 4>>

module {
  func.func @copy_ub_to_ub(
      %src_memref: memref<64xf32, #hivm.address_space<ub>>,
      %dst_memref: memref<64xf32, #hivm.address_space<ub>>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c64 = arith.constant 64 : index
    %src = tla.tensor_desc %src_memref shape [%c1, %c64, %c1, %c1] stride [%c64, %c1, %c1, %c1] origin_shape [%c1, %c64] coord [%c0, %c0] : memref<64xf32, #hivm.address_space<ub>> -> !t
    %dst = tla.tensor_desc %dst_memref shape [%c1, %c64, %c1, %c1] stride [%c64, %c1, %c1, %c1] origin_shape [%c1, %c64] coord [%c0, %c0] : memref<64xf32, #hivm.address_space<ub>> -> !t
    "tla.vector"() ({
      "tla.copy"(%dst, %src) : (!t, !t) -> ()
    }) : () -> ()
    return
  }
}
