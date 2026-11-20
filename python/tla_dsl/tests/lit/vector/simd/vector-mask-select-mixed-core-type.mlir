// RUN: %tla_compile %s -o %t
// RUN: %filecheck %s < %t

// HIVMC needs an explicit core type for the mask-select bitcode in a MIX kernel.

!gm = !tla.tensor<!tla.layout<!tla.shape<32>, !tla.stride<1>, !tla.shape<32>, RowMajor>, !tla.coord<0>, !tla.ptr<i8, gm, 32>>
!ub = !tla.tensor<!tla.layout<!tla.shape<32>, !tla.stride<1>, !tla.shape<32>, RowMajor>, !tla.coord<0>, !tla.ptr<i8, ub, 32>>
!mask = !tla.mask<256>

module {
  tla.func @mixed_mask_select(%output: !gm) {
    "tla.cube"() ({
    }) : () -> ()
    "tla.vector"() ({
      %buffer = tla.alloc_ptr{size_bytes = 32} -> !tla.ptr<i8, ub, 32>
      %bits = "tla.make_tensor_like"(%buffer, %output) {layoutTag = #tla.layout_tag<RowMajor>} : (!tla.ptr<i8, ub, 32>, !gm) -> !ub
      "tla.vec.func"() ({
        %selector = "tla.create_mask"() {pattern = "H", dtype = i8} : () -> !mask
        %x = "tla.create_mask"() {pattern = "Q", dtype = i8} : () -> !mask
        %y = "tla.create_mask"() {pattern = "M4", dtype = i8} : () -> !mask
        %result = "tla.where"(%selector, %x, %y) : (!mask, !mask, !mask) -> !mask
        tla.store %bits, %result : !ub, !mask
      }) {mode = "simd"} : () -> ()
      tla.copy %output, %bits : !gm, !ub
    }) : () -> ()
    tla.return
  }
}

// CHECK: func.func private @tla_mask_select(
// CHECK-SAME: hivm.func_core_type = #hivm.func_core_type<AIV>
// CHECK-LABEL: func.func @mixed_mask_select_mix_aic(
// CHECK-LABEL: func.func @mixed_mask_select_mix_aiv(
// CHECK: call @vector_region_
// CHECK-LABEL: func.func private @vector_region_
// CHECK: call @tla_mask_select(
