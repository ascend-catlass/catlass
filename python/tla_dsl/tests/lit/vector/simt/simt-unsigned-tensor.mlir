// RUN: %tla_compile %s --mlir-print-ir-after=tla-vector-region-simt -o %t 2>&1 | %filecheck %s

// Unsigned storage boundaries preserve signless SIMT arithmetic.

!fscalar = !tla.tensor<!tla.layout<!tla.shape<1>, !tla.stride<1>, !tla.shape<1>, RowMajor>, !tla.coord<0>, !tla.ptr<ui32, gm, 4>>

module {
  func.func @simt_unsigned_tensor(%scalar: ui32,
      %src_memref: memref<1xui32, #hivm.address_space<gm>>,
      %dst_memref: memref<1xui32, #hivm.address_space<gm>>) {
    %src_c0 = arith.constant 0 : index
    %src_c1 = arith.constant 1 : index
    %src = tla.tensor_desc %src_memref shape [%src_c1, %src_c1, %src_c1, %src_c1] stride [%src_c1, %src_c1, %src_c1, %src_c1] origin_shape [%src_c1, %src_c1] coord [%src_c0, %src_c0] : memref<1xui32, #hivm.address_space<gm>> -> !fscalar
    %dst_c0 = arith.constant 0 : index
    %dst_c1 = arith.constant 1 : index
    %dst = tla.tensor_desc %dst_memref shape [%dst_c1, %dst_c1, %dst_c1, %dst_c1] stride [%dst_c1, %dst_c1, %dst_c1, %dst_c1] origin_shape [%dst_c1, %dst_c1] coord [%dst_c0, %dst_c0] : memref<1xui32, #hivm.address_space<gm>> -> !fscalar
    %captured = memref.load %src_memref[%src_c0] : memref<1xui32, #hivm.address_space<gm>>
    "tla.vec.func"() ({
      %i = arith.constant 0 : index
      %x = tla.simt_load %src[%i] : <!tla.layout<!tla.shape<1>, !tla.stride<1>, !tla.shape<1>, RowMajor>, !tla.coord<0>, !tla.ptr<ui32, gm, 4>> -> ui32
      %value = builtin.unrealized_conversion_cast %x : ui32 to i32
      %arg = builtin.unrealized_conversion_cast %scalar : ui32 to i32
      %outer = builtin.unrealized_conversion_cast %captured : ui32 to i32
      %partial = tla.simt_add %value, %arg : i32
      %sum = tla.simt_add %partial, %outer : i32
      %ab = builtin.unrealized_conversion_cast %sum : i32 to ui32
      tla.simt_store %dst[%i], %ab : <!tla.layout<!tla.shape<1>, !tla.stride<1>, !tla.shape<1>, RowMajor>, !tla.coord<0>, !tla.ptr<ui32, gm, 4>>, ui32
    }) {mode = "simt", thread_block_dim = array<i64: 1, 1, 1>} : () -> ()
    return
  }
}

// CHECK: hivm_regbaseintrins.intrins.launch_func @simt_unsigned_tensor_vf_simt
// CHECK-LABEL: func.func @simt_unsigned_tensor_vf_simt
// CHECK: memref.load
// CHECK: arith.addi
// CHECK: memref.store
