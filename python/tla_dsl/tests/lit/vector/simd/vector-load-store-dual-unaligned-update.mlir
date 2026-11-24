// RUN: %tla_compile %s --mlir-print-ir-after=tla-unalign-ureg -o %t 2>&1 | %filecheck %s --check-prefix=STATE --implicit-check-not=tla.local_mem_bar
// RUN: %tla_compile %s --mlir-print-ir-after=tla-vector-region-simd -o %t 2>&1 | %filecheck %s --check-prefix=LOWER

!tile = !tla.tensor<!tla.layout<!tla.shape<256>, !tla.stride<1>, !tla.shape<256>, RowMajor>, !tla.coord<0>, !tla.ptr<i16, ub, 2>>
!reg = !tla.vector<128xi16>
!ptr = !tla.ptr<i16, ub, 2>

module {
  func.func @dual_unaligned(%trips: index, %cond: i1, %stride: i32) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c256 = arith.constant 256 : index
    %input = tla.alloc_ptr{size_bytes = 1536} -> !ptr
    %output = tla.alloc_ptr{size_bytes = 1536} -> !ptr
    %src = tla.tensor_desc %input shape [%c1, %c256, %c1, %c1] stride [%c256, %c1, %c1, %c1] origin_shape [%c1, %c256] coord [%c0, %c0] : !ptr -> !tile
    %dst = tla.tensor_desc %output shape [%c1, %c256, %c1, %c1] stride [%c256, %c1, %c1, %c1] origin_shape [%c1, %c256] coord [%c0, %c0] : !ptr -> !tile
    "tla.vec.func"() ({
      %windows:2 = scf.for %i = %c0 to %trips step %c1 iter_args(%in = %src, %out = %dst) -> (!tile, !tile) {
        %even, %odd, %sp, %lu = tla.load_unalign_update %in post_update_stride %c256 : !tile post_update_stride index -> !reg second !reg, !ptr, !tla.ureg
        %dp, %su = tla.store_unalign_update %out, %odd, %even post_update_stride %c256 : !tile, !reg, !reg post_update_stride index -> !ptr, !tla.ureg
        %next_in = tla.make_tensor_like %sp like %src layoutTag(<RowMajor>) : !ptr, !tile -> !tile
        %next_out = tla.make_tensor_like %dp like %dst layoutTag(<RowMajor>) : !ptr, !tile -> !tile
        tla.local_mem_bar 2
        scf.yield %next_in, %next_out : !tile, !tile
      }
      scf.if %cond {
        %even, %odd, %sp, %lu = tla.load_unalign_update %src post_update_stride %stride : !tile post_update_stride i32 -> !reg second !reg, !ptr, !tla.ureg
        %dp, %su = tla.store_unalign_update %dst, %odd, %even post_update_stride %stride : !tile, !reg, !reg post_update_stride i32 -> !ptr, !tla.ureg
        %next_out = tla.make_tensor_like %dp like %dst layoutTag(<RowMajor>) : !ptr, !tile -> !tile
      }
    }) {mode = "simd"} : () -> ()
    return
  }
}

// STATE-DAG: tla.load_unalign_pre
// STATE-DAG: tla.create_ureg
// STATE: scf.for {{.*}}!tla.ureg{{.*}}!tla.ureg
// STATE: tla.load_unalign_update {{.*}} ureg_in
// STATE: tla.store_unalign_update {{.*}} ureg_in
// STATE: tla.local_mem_bar 2
// STATE: scf.yield
// STATE: tla.store_unalign_post
// STATE: scf.if
// STATE: tla.load_unalign_pre
// STATE: tla.load_unalign_update {{.*}} ureg_in
// STATE: tla.create_ureg
// STATE: tla.store_unalign_update {{.*}} ureg_in
// STATE: tla.store_unalign_post

// LOWER: scf.for
// LOWER-COUNT-2: hivm_regbaseintrins.intr.hivm.vldus.post.s16
// LOWER: ave.hir.vdintlv
// LOWER: ave.hir.vintlv
// LOWER-COUNT-2: hivm_regbaseintrins.intr.hivm.vstus.post.s16
// LOWER: ave.hir.membar
// LOWER: scf.yield
// LOWER: hivm_regbaseintrins.intr.hivm.vstas
