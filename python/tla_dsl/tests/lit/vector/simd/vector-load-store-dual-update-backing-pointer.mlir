// RUN: %tla_compile %s --mlir-print-ir-after=tla-vector-region-simd -o %t 2>&1 | %filecheck %s --check-prefix=ALIGNED
// RUN: sed 's/%offset = arith.constant 0 : index/%offset = arith.constant 15 : index/' %s > %t.misaligned.mlir
// RUN: not %tla_compile %t.misaligned.mlir -o %t 2>&1 | %filecheck %s --check-prefix=MISALIGNED

!ptr = !tla.ptr<i16, ub, 32>
!src = !tla.tensor<!tla.layout<!tla.shape<256>, !tla.stride<1>, !tla.shape<256>, RowMajor>, !tla.coord<0>, !ptr>
!view = !tla.tensor<!tla.layout<!tla.shape<256>, !tla.stride<1>, !tla.shape<256>, RowMajor>, !tla.coord<1>, !ptr>
!dst = !tla.tensor<!tla.layout<!tla.shape<128>, !tla.stride<1>, !tla.shape<128>, RowMajor>, !tla.coord<0>, !ptr>
!reg = !tla.vector<128xi16>
module {
  func.func @load_from_view_backing() {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c128 = arith.constant 128 : index
    %c256 = arith.constant 256 : index
    %offset = arith.constant 0 : index
    %alloc = tla.alloc_ptr{size_bytes = 1056} -> !ptr
    %base = tla.ptr_add %alloc, %offset : !ptr, index -> !ptr
    %out = tla.alloc_ptr{size_bytes = 256} -> !ptr
    %view = tla.tensor_desc %base shape [%c1, %c256, %c1, %c1] stride [%c256, %c1, %c1, %c1] origin_shape [%c1, %c256] coord [%c0, %c1] : !ptr -> !view
    %ptr = tla.tensor_ptr %view : !view -> !ptr
    %src = tla.tensor_desc %ptr shape [%c1, %c256, %c1, %c1] stride [%c256, %c1, %c1, %c1] origin_shape [%c1, %c256] coord [%c0, %c0] : !ptr -> !src
    %dst = tla.tensor_desc %out shape [%c1, %c128, %c1, %c1] stride [%c128, %c1, %c1, %c1] origin_shape [%c1, %c128] coord [%c0, %c0] : !ptr -> !dst
    "tla.vec.func"() ({
      %even, %odd, %next = tla.load_update %src post_update_stride %c256 {load_dist = #tla.load_dist<dintlv_b16>} : !src post_update_stride index -> !reg second !reg, !ptr
      tla.store %dst, %odd : !dst, !reg
    }) : () -> ()
    return
  }
}

// ALIGNED: hivm_regbaseintrins.intr.hivm.vldsx2.post.v128u16
// MISALIGNED: 'tla.load_update' op dual UPDATE effective UB address must be 32-byte aligned
