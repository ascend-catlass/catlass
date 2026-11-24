// RUN: %tla_compile %s --mlir-print-ir-after=tla-vector-region-simd -o %t 2>&1 | %filecheck %s
// RUN: sed 's/post_update_stride %c8 :/post_update_stride %c8 {compressed = true} :/' %s | not %tla_compile - -o %t 2>&1 | %filecheck %s --check-prefix=COMPRESSED

!dst16 = !tla.tensor<!tla.layout<!tla.shape<8>, !tla.stride<1>, !tla.shape<8>, RowMajor>, !tla.coord<0>, !tla.ptr<f16, ub, 32>>
!dst32 = !tla.tensor<!tla.layout<!tla.shape<2>, !tla.stride<1>, !tla.shape<2>, RowMajor>, !tla.coord<0>, !tla.ptr<i32, ub, 32>>

module {
  func.func @store_mask_b16() {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c8 = arith.constant 8 : index
    %buffer = tla.alloc_ptr {size_bytes = 32} -> !tla.ptr<f16, ub, 32>
    %dst = tla.tensor_desc %buffer shape [%c1, %c8, %c1, %c1] stride [%c8, %c1, %c1, %c1] origin_shape [%c1, %c8] coord [%c0, %c0] : !tla.ptr<f16, ub, 32> -> !dst16
    "tla.vec.func"() ({
      %mask = "tla.create_mask"() {pattern = "VL1", dtype = f16} : () -> !tla.mask<128>
      %ptr, %ureg = tla.store_unalign_update %dst, %mask post_update_stride %c8 : !dst16, !tla.mask<128> post_update_stride index -> !tla.ptr<f16, ub, 32>, !tla.ureg
      %next = tla.make_tensor_like %ptr like %dst layoutTag(<RowMajor>) : !tla.ptr<f16, ub, 32>, !dst16 -> !dst16
    }) : () -> ()
    return
  }

  func.func @store_mask_b32_stream() {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %trips = arith.constant 3 : index
    %buffer = tla.alloc_ptr {size_bytes = 32} -> !tla.ptr<i32, ub, 32>
    %dst = tla.tensor_desc %buffer shape [%c1, %c2, %c1, %c1] stride [%c2, %c1, %c1, %c1] origin_shape [%c1, %c2] coord [%c0, %c0] : !tla.ptr<i32, ub, 32> -> !dst32
    "tla.vec.func"() ({
      %end = scf.for %i = %c0 to %trips step %c1 iter_args(%window = %dst) -> (!dst32) {
        %mask, %rem = tla.update_mask %i, i32 : !tla.mask<64>, index
        %ptr, %ureg = tla.store_unalign_update %window, %mask post_update_stride %c2 : !dst32, !tla.mask<64> post_update_stride index -> !tla.ptr<i32, ub, 32>, !tla.ureg
        %next = tla.make_tensor_like %ptr like %window layoutTag(<RowMajor>) : !tla.ptr<i32, ub, 32>, !dst32 -> !dst32
        scf.yield %next : !dst32
      }
    }) : () -> ()
    return
  }
}

// CHECK-LABEL: func.func private @vector_region_
// CHECK: "hivm_regbaseintrins.intr.hivm.init.vector.align.data"
// CHECK: "hivm_regbaseintrins.intr.hivm.pstu.b16"
// CHECK: "hivm_regbaseintrins.intr.hivm.vstas"

// CHECK-LABEL: func.func private @vector_region_
// CHECK: [[INIT:%.*]] = "hivm_regbaseintrins.intr.hivm.init.vector.align.data"
// CHECK: scf.for {{.*}} iter_args({{.*}}[[UREG:%[a-zA-Z0-9_]+]] = [[INIT]]{{.*}})
// CHECK-NOT: "hivm_regbaseintrins.intr.hivm.init.vector.align.data"
// CHECK-NOT: "hivm_regbaseintrins.intr.hivm.vstas"
// CHECK: [[PSTU:%.*]] = "hivm_regbaseintrins.intr.hivm.pstu.b32"({{.*}}, [[UREG]])
// CHECK: [[NEXT_UREG:%[a-zA-Z0-9_]+]] = llvm.extractvalue [[PSTU]][0]
// CHECK: llvm.extractvalue [[PSTU]][1]
// CHECK-NOT: "hivm_regbaseintrins.intr.hivm.vstas"
// CHECK: scf.yield {{.*}}[[NEXT_UREG]]
// CHECK-NEXT: }
// CHECK: "hivm_regbaseintrins.intr.hivm.vstas"
// CHECK-NOT: "hivm_regbaseintrins.intr.hivm.vstas"

// COMPRESSED: error: 'tla.store_unalign_update' op compressed UPDATE requires a vector source
