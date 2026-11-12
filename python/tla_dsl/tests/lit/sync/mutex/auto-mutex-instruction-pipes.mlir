// RUN: split-file %s %t
// RUN: %tla_compile %t/zero.mlir -o %t/zero.o --mlir-print-ir-after=tla-insert-auto-mutex 2>&1 | %filecheck %s
// RUN: %tla_compile %t/unit-flag.mlir -o %t/unit-flag.o --mlir-print-ir-after=tla-insert-auto-mutex 2>&1 | %filecheck %s --check-prefix=UNIT

//--- zero.mlir

!gm = !tla.tensor<!tla.layout<!tla.shape<32,32>, !tla.stride<32,1>, !tla.shape<32,32>, RowMajor>, !tla.coord<0,0>, !tla.ptr<f32, gm, 4>>
!l1 = !tla.tensor<!tla.layout<!tla.shape<(16,2),(8,4)>, !tla.stride<(8,128),(1,256)>, !tla.shape<32,32>, zN>, !tla.coord<0,0>, !tla.ptr<f32, l1, 512>>
!l0a = !tla.tensor<!tla.layout<!tla.shape<(16,2),(8,4)>, !tla.stride<(8,128),(1,256)>, !tla.shape<32,32>, zN>, !tla.coord<0,0>, !tla.ptr<f32, l0a, 512>>
!l0b = !tla.tensor<!tla.layout<!tla.shape<(8,4),(16,2)>, !tla.stride<(1,256),(8,128)>, !tla.shape<32,32>, nZ>, !tla.coord<0,0>, !tla.ptr<f32, l0b, 512>>
!l0c = !tla.tensor<!tla.layout<!tla.shape<(16,2),(16,2)>, !tla.stride<(16,256),(1,512)>, !tla.shape<32,32>, L0Clayout>, !tla.coord<0,0>, !tla.ptr<f32, l0c, 1024>>
!ub = !tla.tensor<!tla.layout<!tla.shape<32,32>, !tla.stride<32,1>, !tla.shape<32,32>, RowMajor>, !tla.coord<0,0>, !tla.ptr<f32, ub, 256>>

module {
  tla.func @auto_mutex_instruction_pipes(%gm: !gm) attributes {tla.auto_sync = "v0"} {
    // Deliberately allocate out of physical resource order. Mutex IDs are
    // assigned by L1 < L0A < L0B < L0C < UB, not by source order.
    %l0c_ptr = tla.alloc_ptr{size_bytes = 4096} -> !tla.ptr<f32, l0c, 1024>
    %l0b_ptr = tla.alloc_ptr{size_bytes = 4096} -> !tla.ptr<f32, l0b, 512>
    %l0a_ptr = tla.alloc_ptr{size_bytes = 4096} -> !tla.ptr<f32, l0a, 512>
    %l1_ptr = tla.alloc_ptr{size_bytes = 4096} -> !tla.ptr<f32, l1, 512>
    %ub_ptr = tla.alloc_ptr{size_bytes = 4096} -> !tla.ptr<f32, ub, 256>
    %l1 = "tla.make_tensor_like"(%l1_ptr, %gm) {layoutTag = #tla.layout_tag<zN>} : (!tla.ptr<f32, l1, 512>, !gm) -> !l1
    %l0a = "tla.make_tensor_like"(%l0a_ptr, %l1) {layoutTag = #tla.layout_tag<zN>} : (!tla.ptr<f32, l0a, 512>, !l1) -> !l0a
    %l0b = "tla.make_tensor_like"(%l0b_ptr, %gm) {layoutTag = #tla.layout_tag<nZ>} : (!tla.ptr<f32, l0b, 512>, !gm) -> !l0b
    %l0c = "tla.make_tensor_like"(%l0c_ptr, %gm) {layoutTag = #tla.layout_tag<L0Clayout>} : (!tla.ptr<f32, l0c, 1024>, !gm) -> !l0c
    %ub = "tla.make_tensor_like"(%ub_ptr, %gm) {layoutTag = #tla.layout_tag<RowMajor>} : (!tla.ptr<f32, ub, 256>, !gm) -> !ub
    %params = "tla.CopyL0C2DstParams"() <{unit_flag = 0 : i64, relu_enable = false, quant_mode = #tla.quant_mode<NO_QUANT>, l0c2ub_mode = #tla.l0c2ub_mode<NO_SPLIT_VEC_0>}> : () -> !tla.copy_l0c2dst_params
    %init = arith.constant true
    %unit = arith.constant 0 : i64
    "tla.cube"() ({
      tla.copy %l1, %gm : !l1, !gm
      tla.copy %l0a, %l1 : !l0a, !l1
      tla.mmad %l0c, %l0a, %l0b, %init, %unit {compute_order = #tla.compute_order<M_FIRST>, hf32_mode = #tla.hf32_mode<HF32_DISABLE>} : !l0c, !l0a, !l0b, i1, i64
      "tla.copy"(%ub, %l0c, %params) : (!ub, !l0c, !tla.copy_l0c2dst_params) -> ()
    }) : () -> ()
    "tla.vector"() ({
      tla.copy %ub, %gm : !ub, !gm
      tla.copy %gm, %ub : !gm, !ub
    }) : () -> ()
    tla.return
  }
}

// One compact kernel covers all supported instruction classes and pipe
// inference, physical ID order, stack order, and independent core-side IDs.
// CHECK-LABEL: func.func @auto_mutex_instruction_pipes
// CHECK-NOT: tla.auto_sync
// CHECK: %[[L1:.*]] = tla.mutex "auto_l1_0_4096" {id = 0 : i64} -> !tla.mutex
// CHECK-NEXT: %[[L0A:.*]] = tla.mutex "auto_l0a_0_4096" {id = 1 : i64} -> !tla.mutex
// CHECK-NEXT: %[[L0B:.*]] = tla.mutex "auto_l0b_0_4096" {id = 2 : i64} -> !tla.mutex
// CHECK-NEXT: %[[L0C:.*]] = tla.mutex "auto_l0c_0_4096" {id = 3 : i64} -> !tla.mutex
// CHECK-NEXT: %[[UB_CUBE:.*]] = tla.mutex "auto_ub_0_4096" {id = 4 : i64} -> !tla.mutex
// CHECK-NEXT: %[[UB_VECTOR:.*]] = tla.mutex "auto_ub_0_4096" {id = 0 : i64} -> !tla.mutex
// MTE2: GM -> L1.
// CHECK: tla.mutex_lock %[[L1]][<mte2>] : !tla.mutex
// CHECK-NEXT: tla.copy
// CHECK-NEXT: tla.mutex_unlock %[[L1]][<mte2>] : !tla.mutex
// MTE1: L1 -> L0A, with stack order.
// CHECK: tla.mutex_lock %[[L1]][<mte1>] : !tla.mutex
// CHECK-NEXT: tla.mutex_lock %[[L0A]][<mte1>] : !tla.mutex
// CHECK-NEXT: tla.copy
// CHECK-NEXT: tla.mutex_unlock %[[L0A]][<mte1>] : !tla.mutex
// CHECK-NEXT: tla.mutex_unlock %[[L1]][<mte1>] : !tla.mutex
// CUBE: MMAD locks L0A/L0B/L0C in physical order and unlocks in reverse.
// CHECK: tla.mutex_lock %[[L0A]][<cube>] : !tla.mutex
// CHECK-NEXT: tla.mutex_lock %[[L0B]][<cube>] : !tla.mutex
// CHECK-NEXT: tla.mutex_lock %[[L0C]][<cube>] : !tla.mutex
// CHECK-NEXT: tla.mmad
// CHECK-NEXT: tla.mutex_unlock %[[L0C]][<cube>] : !tla.mutex
// CHECK-NEXT: tla.mutex_unlock %[[L0B]][<cube>] : !tla.mutex
// CHECK-NEXT: tla.mutex_unlock %[[L0A]][<cube>] : !tla.mutex
// FIX: L0C -> UB, still using the Cube-side UB mutex.
// CHECK: tla.mutex_lock %[[L0C]][<fix>] : !tla.mutex
// CHECK-NEXT: tla.mutex_lock %[[UB_CUBE]][<fix>] : !tla.mutex
// CHECK-NEXT: tla.copy
// CHECK-NEXT: tla.mutex_unlock %[[UB_CUBE]][<fix>] : !tla.mutex
// CHECK-NEXT: tla.mutex_unlock %[[L0C]][<fix>] : !tla.mutex
// The Vector side owns a distinct ID-0 mutex for both MTE2 and MTE3.
// CHECK: tla.mutex_lock %[[UB_VECTOR]][<mte2>] : !tla.mutex
// CHECK-NEXT: tla.copy
// CHECK-NEXT: tla.mutex_unlock %[[UB_VECTOR]][<mte2>] : !tla.mutex
// CHECK-NEXT: tla.mutex_lock %[[UB_VECTOR]][<mte3>] : !tla.mutex
// CHECK-NEXT: tla.copy
// CHECK-NEXT: tla.mutex_unlock %[[UB_VECTOR]][<mte3>] : !tla.mutex

//--- unit-flag.mlir
!unit_gm = !tla.tensor<!tla.layout<!tla.shape<32,32>, !tla.stride<32,1>, !tla.shape<32,32>, RowMajor>, !tla.coord<0,0>, !tla.ptr<f32, gm, 4>>
!unit_l0a = !tla.tensor<!tla.layout<!tla.shape<(16,2),(8,4)>, !tla.stride<(8,128),(1,256)>, !tla.shape<32,32>, zN>, !tla.coord<0,0>, !tla.ptr<f32, l0a, 512>>
!unit_l0b = !tla.tensor<!tla.layout<!tla.shape<(8,4),(16,2)>, !tla.stride<(1,256),(8,128)>, !tla.shape<32,32>, nZ>, !tla.coord<0,0>, !tla.ptr<f32, l0b, 512>>
!unit_l0c = !tla.tensor<!tla.layout<!tla.shape<(16,2),(16,2)>, !tla.stride<(16,256),(1,512)>, !tla.shape<32,32>, L0Clayout>, !tla.coord<0,0>, !tla.ptr<f32, l0c, 1024>>
!unit_ub = !tla.tensor<!tla.layout<!tla.shape<32,32>, !tla.stride<32,1>, !tla.shape<32,32>, RowMajor>, !tla.coord<0,0>, !tla.ptr<f32, ub, 256>>

module {
  tla.func @auto_mutex_unit_flag(%cond: i1, %gm: !unit_gm) attributes {tla.auto_sync = "v0"} {
    %pa = tla.alloc_ptr{size_bytes = 4096} -> !tla.ptr<f32, l0a, 512>
    %pb = tla.alloc_ptr{size_bytes = 4096} -> !tla.ptr<f32, l0b, 512>
    %pc = tla.alloc_ptr{size_bytes = 4096} -> !tla.ptr<f32, l0c, 1024>
    %pub = tla.alloc_ptr{size_bytes = 4096} -> !tla.ptr<f32, ub, 256>
    %a = "tla.make_tensor_like"(%pa, %gm) {layoutTag = #tla.layout_tag<zN>} : (!tla.ptr<f32, l0a, 512>, !unit_gm) -> !unit_l0a
    %b = "tla.make_tensor_like"(%pb, %gm) {layoutTag = #tla.layout_tag<nZ>} : (!tla.ptr<f32, l0b, 512>, !unit_gm) -> !unit_l0b
    %c = "tla.make_tensor_like"(%pc, %gm) {layoutTag = #tla.layout_tag<L0Clayout>} : (!tla.ptr<f32, l0c, 1024>, !unit_gm) -> !unit_l0c
    %ub = "tla.make_tensor_like"(%pub, %gm) {layoutTag = #tla.layout_tag<RowMajor>} : (!tla.ptr<f32, ub, 256>, !unit_gm) -> !unit_ub
    %init = arith.constant true
    %unit32 = scf.if %cond -> (i32) {
      %two = arith.constant 2 : i32
      scf.yield %two : i32
    } else {
      %three = arith.constant 3 : i32
      scf.yield %three : i32
    }
    %unit64 = arith.extsi %unit32 : i32 to i64
    %params = "tla.CopyL0C2DstParams"() <{unit_flag = 3 : i64, relu_enable = false, quant_mode = #tla.quant_mode<NO_QUANT>, l0c2ub_mode = #tla.l0c2ub_mode<NO_SPLIT_VEC_0>}> : () -> !tla.copy_l0c2dst_params
    "tla.cube"() ({
      tla.mmad %c, %a, %b, %init, %unit64 {compute_order = #tla.compute_order<M_FIRST>, hf32_mode = #tla.hf32_mode<HF32_DISABLE>} : !unit_l0c, !unit_l0a, !unit_l0b, i1, i64
      "tla.copy"(%ub, %c, %params) : (!unit_ub, !unit_l0c, !tla.copy_l0c2dst_params) -> ()
    }) : () -> ()
    tla.return
  }
}

// A provably enabled MMAD unit flag (2 or 3) and FIX unit flag 3 make L0C
// synchronization part of the unit-flag protocol. L0A/L0B and UB remain
// independently protected by mutexes.
// UNIT-LABEL: func.func @auto_mutex_unit_flag
// UNIT: %[[UNIT_A:.*]] = tla.mutex "auto_l0a_0_4096" {id = 0 : i64} -> !tla.mutex
// UNIT-NEXT: %[[UNIT_B:.*]] = tla.mutex "auto_l0b_0_4096" {id = 1 : i64} -> !tla.mutex
// UNIT-NEXT: %[[UNIT_UB:.*]] = tla.mutex "auto_ub_0_4096" {id = 2 : i64} -> !tla.mutex
// UNIT: tla.mutex_lock %[[UNIT_A]][<cube>] : !tla.mutex
// UNIT-NEXT: tla.mutex_lock %[[UNIT_B]][<cube>] : !tla.mutex
// UNIT-NEXT: tla.mmad
// UNIT-NEXT: tla.mutex_unlock %[[UNIT_B]][<cube>] : !tla.mutex
// UNIT-NEXT: tla.mutex_unlock %[[UNIT_A]][<cube>] : !tla.mutex
// UNIT-NEXT: tla.mutex_lock %[[UNIT_UB]][<fix>] : !tla.mutex
// UNIT-NEXT: tla.copy
// UNIT-NEXT: tla.mutex_unlock %[[UNIT_UB]][<fix>] : !tla.mutex
