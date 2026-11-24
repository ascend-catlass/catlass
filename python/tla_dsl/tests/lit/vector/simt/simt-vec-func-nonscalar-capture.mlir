// RUN: not %tla_compile %s -o %t 2>&1 | %filecheck %s

// A SIMT region that indexes a raw memref directly. This used to be rejected
// outright; when raw-pointer captures were added the rule was briefly widened
// to accept any memref defined outside the region, which flattened a strided or
// dynamic-stride buffer to an identity layout and could silently read the wrong
// element. Only the t.ptr address read (extract_aligned_pointer_as_index) is
// layout-free, so direct indexing stays rejected -- see
// simt-tensor-ptr-raw-capture-strided.mlir for the accepted pointer form.

module {
  func.func @simt_nonscalar_capture(%raw: memref<128xf32, strided<[2]>, #hivm.address_space<gm>>) {
    %c0 = arith.constant 0 : index
    "tla.vector"() ({
      "tla.vec.func"() ({
        %v = memref.load %raw[%c0] : memref<128xf32, strided<[2]>, #hivm.address_space<gm>>
        %s = tla.simt_add %v, %v : f32
        memref.store %s, %raw[%c0] : memref<128xf32, strided<[2]>, #hivm.address_space<gm>>
      }) {mode = "simt", thread_block_dim = array<i64: 64, 1, 1>} : () -> ()
    }) : () -> ()
    return
  }
}

// The region is diagnosed on the user-visible op, naming the offending buffer.
// CHECK: error: 'tla.vec.func' op a SIMT tla.vec.func can only read a buffer through its tensor
// CHECK-SAME: memref<128xf32, strided<[2]>, #hivm.address_space<gm>>
