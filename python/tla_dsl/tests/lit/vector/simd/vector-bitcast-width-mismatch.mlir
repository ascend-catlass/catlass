// RUN: not %tla_compile %s -o - 2>&1 | %filecheck %s

// A register view must preserve the valid bit count. 64xf32 can be viewed as
// 128xf16, but 64xf16 would discard half of the valid source bits.
// CHECK: 'tla.bitcast' op preserves the valid bit count

!fvec = !tla.tensor<!tla.layout<!tla.shape<64>, !tla.stride<1>, !tla.shape<64>, RowMajor>, !tla.coord<0>, !tla.ptr<f32, ub, 4>>
!hvec = !tla.tensor<!tla.layout<!tla.shape<64>, !tla.stride<1>, !tla.shape<64>, RowMajor>, !tla.coord<0>, !tla.ptr<f16, ub, 2>>

module {
  func.func @vector_bitcast_width_mismatch(
      %src_memref: memref<64xf32, #hivm.address_space<ub>>,
      %dst_memref: memref<64xf16, #hivm.address_space<ub>>) {
    %src = builtin.unrealized_conversion_cast %src_memref : memref<64xf32, #hivm.address_space<ub>> to !fvec
    %dst = builtin.unrealized_conversion_cast %dst_memref : memref<64xf16, #hivm.address_space<ub>> to !hvec
    "tla.vec.func"() ({
      %shape = "tla.make_shape"() : () -> !tla.shape<64>
      %coord = "tla.make_coord"() : () -> !tla.coord<0>
      %src_tile = "tla.tile_view"(%src, %shape, %coord) : (!fvec, !tla.shape<64>, !tla.coord<0>) -> !fvec
      %dst_tile = "tla.tile_view"(%dst, %shape, %coord) : (!hvec, !tla.shape<64>, !tla.coord<0>) -> !hvec
      %v = tla.load %src_tile : !fvec -> !tla.vector<64xf32>
      %bits = tla.bitcast %v : !tla.vector<64xf32> -> !tla.vector<64xf16>
      tla.store %dst_tile, %bits : !hvec, !tla.vector<64xf16>
    }) : () -> ()
    return
  }
}
