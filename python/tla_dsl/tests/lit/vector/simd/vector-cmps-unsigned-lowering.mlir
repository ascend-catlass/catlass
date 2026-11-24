// RUN: %tla_compile %s --mlir-print-ir-after=tla-lower-ave-to-regbase-intrins -o %t 2>&1 | %filecheck %s

!u8vec = !tla.tensor<!tla.layout<!tla.shape<256>, !tla.stride<1>, !tla.shape<256>, RowMajor>, !tla.coord<0>, !tla.ptr<ui8, ub, 1>>

module {
  func.func @unsigned_cmp_and_cmps(
      %lhs_memref: memref<256xui8, #hivm.address_space<ub>>,
      %rhs_memref: memref<256xui8, #hivm.address_space<ub>>,
      %scalar: i8) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c256 = arith.constant 256 : index
    %lhs = tla.tensor_desc %lhs_memref shape [%c1, %c256, %c1, %c1] stride [%c256, %c1, %c1, %c1] origin_shape [%c1, %c256] coord [%c0, %c0] : memref<256xui8, #hivm.address_space<ub>> -> !u8vec
    %rhs = tla.tensor_desc %rhs_memref shape [%c1, %c256, %c1, %c1] stride [%c256, %c1, %c1, %c1] origin_shape [%c1, %c256] coord [%c0, %c0] : memref<256xui8, #hivm.address_space<ub>> -> !u8vec
    "tla.vec.func"() ({
      %a = tla.load %lhs : !u8vec -> !tla.vector<256xui8>
      %b = tla.load %rhs : !u8vec -> !tla.vector<256xui8>
      %vv = tla.cmp "lt" %a, %b : !tla.vector<256xui8>, !tla.vector<256xui8> -> !tla.mask<256>
      %vs = tla.cmp "lt" %a, %scalar : !tla.vector<256xui8>, i8 -> !tla.mask<256>
      tla.store %lhs, %a mask %vv : !u8vec, !tla.vector<256xui8> mask !tla.mask<256>
      tla.store %lhs, %b mask %vs : !u8vec, !tla.vector<256xui8> mask !tla.mask<256>
    }) {mode = "simd"} : () -> ()
    return
  }
}

// CHECK: hivm.vcmp.lt.u.z
// CHECK: hivm.vcmps.lt.u.z
