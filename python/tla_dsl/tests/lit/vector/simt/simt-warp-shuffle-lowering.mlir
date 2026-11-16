// RUN: %tla_compile %s --mlir-print-ir-after=tla-vector-region-simt -o %t 2>&1 | %filecheck %s

// tla.arch.shuffle lowers onto the hivm_regbaseintrins shfl intrinsics: one
// intrinsic family per mode, one overload per value type. The control word is
// built once per op as (index & 31) | (clamp << 8) | ((32 - width) << 16), with
// clamp = 0 for "up" and 0x1f for the other three.

!fscalar = !tla.tensor<!tla.layout<!tla.shape<1>, !tla.stride<1>, !tla.shape<1>, RowMajor>, !tla.coord<0>, !tla.ptr<f32, gm, 4>>

module {
  func.func @warp_shuffle(
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
      %lane = arith.constant 1 : i32
      %x = tla.simt_load %src[%i] : <!tla.layout<!tla.shape<1>, !tla.stride<1>, !tla.shape<1>, RowMajor>, !tla.coord<0>, !tla.ptr<f32, gm, 4>> -> f32

      // All four modes on the native f32 overload.
      %a = "tla.arch.shuffle"(%x, %lane) <{mode = #tla.shuffle_mode<idx>, width = 32 : i32}> : (f32, i32) -> f32
      %b = "tla.arch.shuffle"(%a, %lane) <{mode = #tla.shuffle_mode<up>, width = 32 : i32}> : (f32, i32) -> f32
      %c = "tla.arch.shuffle"(%b, %lane) <{mode = #tla.shuffle_mode<down>, width = 32 : i32}> : (f32, i32) -> f32
      %d = "tla.arch.shuffle"(%c, %lane) <{mode = #tla.shuffle_mode<bfly>, width = 32 : i32}> : (f32, i32) -> f32

      // A narrower group: width=16 changes only the lane-mask field.
      %e = "tla.arch.shuffle"(%d, %lane) <{mode = #tla.shuffle_mode<idx>, width = 16 : i32}> : (f32, i32) -> f32

      // Narrow integers have no intrinsic of their own and ride the i32 one.
      %ibits = arith.fptosi %e : f32 to i16
      %ishuf = "tla.arch.shuffle"(%ibits, %lane) <{mode = #tla.shuffle_mode<bfly>, width = 32 : i32}> : (i16, i32) -> i16
      %iback = arith.sitofp %ishuf : i16 to f32

      tla.simt_store %dst[%i], %iback : <!tla.layout<!tla.shape<1>, !tla.stride<1>, !tla.shape<1>, RowMajor>, !tla.coord<0>, !tla.ptr<f32, gm, 4>>, f32
    }) {mode = "simt", thread_block_dim = array<i64: 1, 1, 1>} : () -> ()
    return
  }
}

// No tla.arch.shuffle may survive the pass.
// CHECK-NOT: tla.arch.shuffle

// clamp 0x1f and width 32 give (0x1f << 8) = 7936; "up" drops the clamp to 0.
// CHECK: arith.constant 7936 : i32
// CHECK: intr.hivm.shfl.idx.f32
// CHECK: arith.constant 0 : i32
// CHECK: intr.hivm.shfl.up.f32
// CHECK: arith.constant 7936 : i32
// CHECK: intr.hivm.shfl.down.f32
// CHECK: arith.constant 7936 : i32
// CHECK: intr.hivm.shfl.bfly.f32

// width=16 puts (32-16)=16 in the lane-mask field: 7936 | (16 << 16) = 1056512.
// CHECK: arith.constant 1056512 : i32
// CHECK: intr.hivm.shfl.idx.f32

// i16 is widened to i32, shuffled there, then truncated back.
// CHECK: arith.extui{{.*}}i16 to i32
// CHECK: intr.hivm.shfl.bfly.i32
// CHECK: arith.trunci{{.*}}i32 to i16
