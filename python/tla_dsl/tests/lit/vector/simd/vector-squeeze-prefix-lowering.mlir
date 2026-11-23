// RUN: %tla_compile %s --mlir-print-ir-after=tla-unalign-ureg -o %t 2>&1 | %filecheck %s --check-prefix=STATE
// RUN: %tla_compile %s --mlir-print-ir-after=tla-vector-region-simd -o %t 2>&1 | %filecheck %s --check-prefix=LOWER

!tile = !tla.tensor<!tla.layout<!tla.shape<128>, !tla.stride<1>, !tla.shape<128>, RowMajor>, !tla.coord<0>, !tla.ptr<i16, ub, 2>>
!vec = !tla.vector<128xi16>
!ptr = !tla.ptr<i16, ub, 2>

module {
  func.func @initialize_before_compression(
      %src_mem: memref<128xi16, #hivm.address_space<ub>>,
      %dst_mem: memref<128xi16, #hivm.address_space<ub>>,
      %other_mem: memref<128xi16, #hivm.address_space<ub>>, %condition: i1, %trips: index) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c128 = arith.constant 128 : index
    %src = tla.tensor_desc %src_mem shape [%c1, %c128, %c1, %c1] stride [%c128, %c1, %c1, %c1] origin_shape [%c1, %c128] coord [%c0, %c0] : memref<128xi16, #hivm.address_space<ub>> -> !tile
    %dst = tla.tensor_desc %dst_mem shape [%c1, %c128, %c1, %c1] stride [%c128, %c1, %c1, %c1] origin_shape [%c1, %c128] coord [%c0, %c0] : memref<128xi16, #hivm.address_space<ub>> -> !tile
    %other = tla.tensor_desc %other_mem shape [%c1, %c128, %c1, %c1] stride [%c128, %c1, %c1, %c1] origin_shape [%c1, %c128] coord [%c0, %c0] : memref<128xi16, #hivm.address_space<ub>> -> !tile
    "tla.vec.func"() ({
      %values = tla.load %src : !tile -> !vec
      %zero = arith.constant 0 : i16
      %initial = "tla.full"(%zero) : (i16) -> !vec
      %selected = scf.if %condition -> (!vec) {
        scf.yield %values : !vec
      } else {
        scf.yield %initial : !vec
      }
      tla.store %other, %selected : !tile, !vec
      tla.store %dst, %initial : !tile, !vec
      %keep = "tla.create_mask"() {pattern = "VL8", dtype = i16} : () -> !tla.mask<128>
      scf.for %i = %c0 to %trips step %c1 {
        %packed = tla.squeeze %values, %keep {store_reg = true} : !vec, !tla.mask<128> -> !vec
        %next, %state = tla.store_unalign_update %dst, %packed post_update_stride %c0 {compressed = true} : !tile, !vec post_update_stride index -> !ptr, !tla.ureg
      }
    }) {mode = "simd"} : () -> ()
    return
  }
}

// STATE-LABEL: func.func @initialize_before_compression(
// STATE: [[DST:%[^ ]+]] = tla.tensor_desc %arg1
// STATE: [[SELECTED:%[^ ]+]] = scf.if %arg3 -> (!tla.vector<128xi16>)
// STATE: tla.store {{.*}}, [[SELECTED]]
// STATE: tla.store [[DST]],
// STATE: [[LOOP:%[^ ]+]] = scf.for {{.*}} -> (!tla.ureg)
// STATE: tla.store_unalign_update [[DST]], {{.*}} {compressed = true}
// STATE: scf.yield {{.*}} : !tla.ureg
// STATE-NEXT: }
// STATE-NEXT: tla.store_unalign_post [[LOOP]], [[DST]] {compressed = true}

// LOWER: call @squeeze_store_int16_t(
// LOWER: call @squeeze_store_post_uint8_t(
