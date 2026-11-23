// RUN: %tla_compile %s --mlir-print-ir-after=tla-unalign-ureg -o %t 2>&1 | %filecheck %s --check-prefix=STATE
// RUN: %tla_compile %s --mlir-print-ir-after=tla-vector-region-simd -o %t 2>&1 | %filecheck %s

!ptr = !tla.ptr<f32, ub, 256>
!tile = !tla.tensor<!tla.layout<!tla.shape<64>, !tla.stride<1>, !tla.shape<64>, RowMajor>, !tla.coord<0>, !tla.ptr<f32, ub, 256>>
!vec = !tla.vector<64xf32>

module {
  // The caller supplies at most two appends and a read offset of 0 or 64.
  func.func @shared_ub(%trips: index, %read_offset: index, %use_other_window: i1) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c64 = arith.constant 64 : index
    %c128 = arith.constant 128 : index
    %c192 = arith.constant 192 : index
    %c256 = arith.constant 256 : index
    %workspace = "tla.alloc_ptr"() {size_bytes = 1280 : i64} : () -> !ptr
    %read_ptr = tla.ptr_add %workspace, %read_offset : !ptr, index -> !ptr
    %write_ptr = tla.ptr_add %workspace, %c128 : !ptr, index -> !ptr
    %other_ptr = tla.ptr_add %workspace, %c192 : !ptr, index -> !ptr
    %count_ptr = tla.ptr_add %workspace, %c256 : !ptr, index -> !ptr
    %src = tla.tensor_desc %read_ptr shape [%c1, %c64, %c1, %c1] stride [%c64, %c1, %c1, %c1] origin_shape [%c1, %c64] coord [%c0, %c0] : !ptr -> !tile
    %dst = tla.tensor_desc %write_ptr shape [%c1, %c64, %c1, %c1] stride [%c64, %c1, %c1, %c1] origin_shape [%c1, %c64] coord [%c0, %c0] : !ptr -> !tile
    %other = tla.tensor_desc %other_ptr shape [%c1, %c64, %c1, %c1] stride [%c64, %c1, %c1, %c1] origin_shape [%c1, %c64] coord [%c0, %c0] : !ptr -> !tile
    %count = tla.tensor_desc %count_ptr shape [%c1, %c64, %c1, %c1] stride [%c64, %c1, %c1, %c1] origin_shape [%c1, %c64] coord [%c0, %c0] : !ptr -> !tile
    "tla.vec.func"() ({
      %start = scf.if %use_other_window -> (!tile) {
        scf.yield %other : !tile
      } else {
        scf.yield %dst : !tile
      }
      %mask = "tla.create_mask"() {pattern = "VL1", dtype = f32} : () -> !tla.mask<64>
      scf.for %i = %c0 to %trips step %c1 {
        %data = tla.load %src : !tile -> !vec
        %packed = tla.squeeze %data, %mask {store_reg = true} : !vec, !tla.mask<64> -> !vec
        %next, %state = tla.store_unalign_update %start, %packed post_update_stride %c0 {compressed = true} : !tile, !vec post_update_stride index -> !ptr, !tla.ureg
      }
      %zero = arith.constant 0.0 : f32
      %summary = tla.full %zero : f32 -> !vec
      tla.store %count, %summary : !tile, !vec
    }) {mode = "simd"} : () -> ()
    return
  }
}

// STATE: [[START:%[a-zA-Z0-9_]+]] = scf.if
// STATE: scf.for
// STATE: tla.store_unalign_update [[START]],
// STATE: tla.store_unalign_post {{.*}}, [[START]] {compressed = true}

// CHECK: ave.hir.vload <NORM>
// CHECK: call @squeeze_store_float(
// CHECK: call @squeeze_store_post_uint8_t(
// CHECK: ave.hir.masked_store
