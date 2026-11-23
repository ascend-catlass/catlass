// RUN: %tla_compile %s -o - | %filecheck %s

!hvec = !tla.tensor<!tla.layout<!tla.shape<128>, !tla.stride<1>, !tla.shape<128>, RowMajor>, !tla.coord<0>, !tla.ptr<f16, ub, 2>>
!vec = !tla.vector<128xf16>

module {
  func.func @squeeze_add_update(
      %src_memref: memref<128xf16, #hivm.address_space<ub>>,
      %ordinary_memref: memref<128xf16, #hivm.address_space<ub>>,
      %compressed_memref: memref<128xf16, #hivm.address_space<ub>>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c128 = arith.constant 128 : index
    %src = tla.tensor_desc %src_memref shape [%c1, %c128, %c1, %c1] stride [%c128, %c1, %c1, %c1] origin_shape [%c1, %c128] coord [%c0, %c0] : memref<128xf16, #hivm.address_space<ub>> -> !hvec
    %ordinary = tla.tensor_desc %ordinary_memref shape [%c1, %c128, %c1, %c1] stride [%c128, %c1, %c1, %c1] origin_shape [%c1, %c128] coord [%c0, %c0] : memref<128xf16, #hivm.address_space<ub>> -> !hvec
    %compressed = tla.tensor_desc %compressed_memref shape [%c1, %c128, %c1, %c1] stride [%c128, %c1, %c1, %c1] origin_shape [%c1, %c128] coord [%c0, %c0] : memref<128xf16, #hivm.address_space<ub>> -> !hvec
    "tla.vec.func"() ({
      %values = tla.load %src : !hvec -> !vec
      %mask = "tla.create_mask"() {pattern = "VL8", dtype = f16} : () -> !tla.mask<128>
      %pack = "tla.squeeze"(%values, %mask) {store_reg = true} : (!vec, !tla.mask<128>) -> !vec
      %sum = tla.add %pack, %values : !vec, !vec -> !vec
      %ordinary_ptr, %ordinary_state = tla.store_unalign_update %ordinary, %sum post_update_stride %c128 : !hvec, !vec post_update_stride index -> !tla.ptr<f16, ub, 2>, !tla.ureg
      %ordinary_window = tla.make_tensor_like %ordinary_ptr like %ordinary layoutTag (<RowMajor>) : !tla.ptr<f16, ub, 2>, !hvec -> !hvec
      %recorded = "tla.squeeze"(%values, %mask) {store_reg = true} : (!vec, !tla.mask<128>) -> !vec
      %adjusted = tla.add %recorded, %values : !vec, !vec -> !vec
      %compressed_ptr, %compressed_state = tla.store_unalign_update %compressed, %adjusted post_update_stride %c0 {compressed = true} : !hvec, !vec post_update_stride index -> !tla.ptr<f16, ub, 2>, !tla.ureg
    }) {mode = "simd"} : () -> ()
    return
  }
}

// CHECK: call @vsqueeze_half_store_reg(
// CHECK: hivm_regbaseintrins.intr.hivm.vadd
// CHECK: hivm_regbaseintrins.intr.hivm.vstus.post.f16
// CHECK: call @vsqueeze_half_store_reg
// CHECK: hivm_regbaseintrins.intr.hivm.vadd
// CHECK: call @squeeze_store_half
// CHECK: call @squeeze_store_post_uint8_t
