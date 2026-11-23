// RUN: %tla_compile %s --mlir-print-ir-after=tla-unalign-ureg -o %t 2>&1 | %filecheck %s --check-prefix=STATE
// RUN: %tla_compile %s --mlir-print-ir-after=tla-vector-region-simd -o %t 2>&1 | %filecheck %s --check-prefix=LOWER

!hvec = !tla.tensor<!tla.layout<!tla.shape<128>, !tla.stride<1>, !tla.shape<128>, RowMajor>, !tla.coord<0>, !tla.ptr<i16, ub, 2>>

module {
  func.func @compression_update(
      %src_memref: memref<128xi16, #hivm.address_space<ub>>,
      %dst_memref: memref<128xi16, #hivm.address_space<ub>>, %condition: i1, %trips: i32) {
    %src_c0 = arith.constant 0 : index
    %src_c1 = arith.constant 1 : index
    %src_c128 = arith.constant 128 : index
    %src = tla.tensor_desc %src_memref shape [%src_c1, %src_c128, %src_c1, %src_c1] stride [%src_c128, %src_c1, %src_c1, %src_c1] origin_shape [%src_c1, %src_c128] coord [%src_c0, %src_c0] : memref<128xi16, #hivm.address_space<ub>> -> !hvec
    %dst_c0 = arith.constant 0 : index
    %dst_c1 = arith.constant 1 : index
    %dst_c128 = arith.constant 128 : index
    %dst = tla.tensor_desc %dst_memref shape [%dst_c1, %dst_c128, %dst_c1, %dst_c1] stride [%dst_c128, %dst_c1, %dst_c1, %dst_c1] origin_shape [%dst_c1, %dst_c128] coord [%dst_c0, %dst_c0] : memref<128xi16, #hivm.address_space<ub>> -> !hvec
    "tla.vec.func"() ({
      %mask = "tla.create_mask"() {pattern = "VL8", dtype = i16} : () -> !tla.mask<128>
      %v0 = tla.load %src : !hvec -> !tla.vector<128xi16>
      %c0 = arith.constant 0 : index
      %lb = arith.constant 0 : i32
      %step = arith.constant 1 : i32
      scf.for %i = %lb to %trips step %step : i32 {
        %packed = "tla.squeeze"(%v0, %mask) {store_reg = true} : (!tla.vector<128xi16>, !tla.mask<128>) -> !tla.vector<128xi16>
        %next, %state = tla.store_unalign_update %dst, %packed post_update_stride %c0 {compressed = true} : !hvec, !tla.vector<128xi16> post_update_stride index -> !tla.ptr<i16, ub, 2>, !tla.ureg
      }
      scf.if %condition {
        %packed = "tla.squeeze"(%v0, %mask) {store_reg = true} : (!tla.vector<128xi16>, !tla.mask<128>) -> !tla.vector<128xi16>
        %next, %state = tla.store_unalign_update %dst, %packed post_update_stride %c0 {compressed = true} : !hvec, !tla.vector<128xi16> post_update_stride index -> !tla.ptr<i16, ub, 2>, !tla.ureg
      }
      tla.store %src, %v0 : !hvec, !tla.vector<128xi16>
    }) {mode = "simd"} : () -> ()
    return
  }
}

// STATE: [[DST:%[^ ]+]] = tla.tensor_desc %arg1
// STATE: tla.clear_squeeze_state
// STATE: [[INIT:%[^ ]+]] = tla.create_ureg
// STATE: [[LOOP:%[^ ]+]] = scf.for {{.*}} iter_args([[INPUT:%[^ ]+]] = [[INIT]]) -> (!tla.ureg)
// STATE: {{.*}}, [[STATE:%[^ ]+]] = tla.store_unalign_update [[DST]], {{.*}} ureg_in [[INPUT]] {compressed = true}
// STATE: scf.yield [[STATE]] : !tla.ureg
// STATE: [[FINAL:%[^ ]+]] = scf.if {{.*}} -> (!tla.ureg)
// STATE: tla.store_unalign_update [[DST]], {{.*}} ureg_in [[LOOP]] {compressed = true}
// STATE: tla.store_unalign_post [[FINAL]], [[DST]] {compressed = true}
// STATE: tla.store {{.*}}!tla.vector<128xi16>
// The first test skips the loop when it has no iterations. The test at the end
// of the body forwards the store's updated state on both the backedge and exit.
// LOWER: call @squeeze_stream_clear()
// LOWER: [[INITIAL:%[^ ]+]] = {{.*}}init.vector.align.data
// LOWER: [[ENTERS:%[^ ]+]] = arith.cmpi slt
// LOWER: [[AFTER_LOOP:%[^ ]+]] = scf.if [[ENTERS]] -> (vector<32xi8>)
// LOWER: [[LOOP_RESULT:%[^ ]+]]:2 = scf.while
// LOWER: call @vsqueeze_int16_t_store_reg(
// LOWER: [[LAST_STORE:%[^ ]+]] = func.call @squeeze_store_int16_t(
// LOWER: scf.condition({{.*}}) {{.*}}, [[LAST_STORE]] : i32, vector<32xi8>
// LOWER: scf.yield [[LOOP_RESULT]]#1 : vector<32xi8>
// LOWER: scf.yield [[INITIAL]] : vector<32xi8>
// LOWER: [[AFTER_IF:%[^ ]+]] = scf.if
// LOWER: call @squeeze_store_int16_t({{.*}}, [[AFTER_LOOP]])
// LOWER: call @squeeze_store_post_uint8_t({{.*}}, [[AFTER_IF]])
// LOWER: ave.hir.masked_store
// LOWER-NOT: tla.store_unalign_update
