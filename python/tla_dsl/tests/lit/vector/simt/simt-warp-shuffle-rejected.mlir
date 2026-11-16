// RUN: not %tla_compile %s -o - 2>&1 | %filecheck %s --check-prefix=CHECK-UP
// RUN: sed 's/shuffle_mode<up>/shuffle_mode<down>/' %s | not %tla_compile - -o - 2>&1 | %filecheck %s --check-prefix=CHECK-DOWN
// RUN: sed 's/shuffle_mode<up>/shuffle_mode<bfly>/' %s | not %tla_compile - -o - 2>&1 | %filecheck %s --check-prefix=CHECK-BFLY
// RUN: sed 's/shuffle_mode<up>/shuffle_mode<idx>/' %s | %tla_compile - -o /dev/null
//
// Diagnostic: for "up", "down" and "bfly" the index is a distance or a lane
// mask, and the 5-bit control field carries one exactly up to 31. From 32 the
// value stops meaning what the caller wrote -- 33 is masked to 1 and reads a
// neighbour instead of leaving the out-of-group lane its own value -- so that is
// where the rejection sits, independent of width. "idx" is the exception: there
// the index is a lane number and src_lane % width is the documented behaviour,
// so the last RUN line requires the very same module to COMPILE once the mode
// is idx.

!fscalar = !tla.tensor<!tla.layout<!tla.shape<1>, !tla.stride<1>, !tla.shape<1>, RowMajor>, !tla.coord<0>, !tla.ptr<f32, gm, 4>>

module {
  func.func @warp_shuffle_out_of_range(
      %src_memref: memref<1xf32, #hivm.address_space<gm>>,
      %dst_memref: memref<1xf32, #hivm.address_space<gm>>) {
    %src_c0 = arith.constant 0 : index
    %src_c1 = arith.constant 1 : index
    %src = tla.tensor_desc %src_memref shape [%src_c1, %src_c1, %src_c1, %src_c1] stride [%src_c1, %src_c1, %src_c1, %src_c1] origin_shape [%src_c1, %src_c1] coord [%src_c0, %src_c0] : memref<1xf32, #hivm.address_space<gm>> -> !fscalar
    %dst_c0 = arith.constant 0 : index
    %dst_c1 = arith.constant 1 : index
    %dst = tla.tensor_desc %dst_memref shape [%dst_c1, %dst_c1, %dst_c1, %dst_c1] stride [%dst_c1, %dst_c1, %dst_c1, %dst_c1] origin_shape [%dst_c1, %dst_c1] coord [%dst_c0, %dst_c0] : memref<1xf32, #hivm.address_space<gm>> -> !fscalar
    "tla.vec.func"() ({
      %i = arith.constant 0 : index
      %delta = arith.constant 33 : i32
      %x = tla.simt_load %src[%i] : <!tla.layout<!tla.shape<1>, !tla.stride<1>, !tla.shape<1>, RowMajor>, !tla.coord<0>, !tla.ptr<f32, gm, 4>> -> f32
      %y = "tla.arch.shuffle"(%x, %delta) <{mode = #tla.shuffle_mode<up>, width = 32 : i32}> : (f32, i32) -> f32
      tla.simt_store %dst[%i], %y : <!tla.layout<!tla.shape<1>, !tla.stride<1>, !tla.shape<1>, RowMajor>, !tla.coord<0>, !tla.ptr<f32, gm, 4>>, f32
    }) {mode = "simt", thread_block_dim = array<i64: 1, 1, 1>} : () -> ()
    return
  }
}

// CHECK-UP: 'tla.arch.shuffle' op index must be less than 32 for this mode, got 33
// CHECK-DOWN: 'tla.arch.shuffle' op index must be less than 32 for this mode, got 33
// CHECK-BFLY: 'tla.arch.shuffle' op index must be less than 32 for this mode, got 33
