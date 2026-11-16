// RUN: %tla_compile %s -o %t --mlir-print-ir-after=tla-insert-auto-mutex > %t.ir 2>&1
// RUN: %filecheck %s < %t.ir

// Scatter writes must participate in the VF's vector-pipe resource set.
// The source, index storage and destination are distinct allocations; the two
// scatter destinations alias one allocation through ptr_add and tile_view.
// Lock each resource once for the whole VF, with reverse-order unlocks.

!layout = !tla.layout<!tla.shape<64>, !tla.stride<1>, !tla.shape<64>, RowMajor>
!coord = !tla.coord<0>
!data = !tla.tensor<!layout, !coord, !tla.ptr<f32, ub, 256>>
!index = !tla.tensor<!layout, !coord, !tla.ptr<i32, ub, 256>>

module {
  tla.func @auto_mutex_scatter() attributes {tla.auto_sync = "v0"} {
    %shape = tla.make_shape -> !tla.shape<64>
    %stride = tla.make_stride -> !tla.stride<1>
    %layout = tla.make_layout %shape, %stride : !tla.shape<64>, !tla.stride<1> -> !layout
    %coord = tla.make_coord -> !coord
    %zero = arith.constant 0 : index
    %src_ptr = tla.alloc_ptr{size_bytes = 256} -> !tla.ptr<f32, ub, 256>
    %idx_ptr = tla.alloc_ptr{size_bytes = 256} -> !tla.ptr<i32, ub, 256>
    %dst_ptr = tla.alloc_ptr{size_bytes = 1024} -> !tla.ptr<f32, ub, 256>
    %alias_ptr = tla.ptr_add %dst_ptr, %zero : !tla.ptr<f32, ub, 256>, index -> !tla.ptr<f32, ub, 256>
    %src = tla.make_tensor %src_ptr, %layout, %coord : !tla.ptr<f32, ub, 256>, !layout, !coord -> !data
    %idx = tla.make_tensor %idx_ptr, %layout, %coord : !tla.ptr<i32, ub, 256>, !layout, !coord -> !index
    %dst = tla.make_tensor %dst_ptr, %layout, %coord : !tla.ptr<f32, ub, 256>, !layout, !coord -> !data
    %alias = tla.make_tensor %alias_ptr, %layout, %coord : !tla.ptr<f32, ub, 256>, !layout, !coord -> !data
    %tile = tla.tile_view %alias, %shape, %coord : !data, !tla.shape<64>, !coord -> !data
    "tla.vector"() ({
      "tla.vec.func"() ({
        %value = tla.load %src : !data -> !tla.vector<64xf32>
        %signed_indices = tla.load %idx : !index -> !tla.vector<64xi32>
        %indices = tla.bitcast %signed_indices : !tla.vector<64xi32> -> !tla.vector<64xui32>
        tla.scatter %dst, %value, %indices : !data, !tla.vector<64xf32>, !tla.vector<64xui32>
        tla.scatter %tile, %value, %indices : !data, !tla.vector<64xf32>, !tla.vector<64xui32>
      }) {mode = "simd"} : () -> ()
    }) : () -> ()
    tla.return
  }
}

// CHECK-LABEL: func.func @auto_mutex_scatter
// CHECK: %[[SRC:.*]] = tla.mutex "auto_ub_0_256" {id = 0 : i64} -> !tla.mutex
// CHECK-NEXT: %[[IDX:.*]] = tla.mutex "auto_ub_256_256" {id = 1 : i64} -> !tla.mutex
// CHECK-NEXT: %[[DST:.*]] = tla.mutex "auto_ub_512_1024" {id = 2 : i64} -> !tla.mutex
// CHECK-NOT: tla.mutex "
// CHECK: "tla.vector"() ({
// CHECK-NEXT: tla.mutex_lock %[[SRC]][<vector>] : !tla.mutex
// CHECK-NEXT: tla.mutex_lock %[[IDX]][<vector>] : !tla.mutex
// CHECK-NEXT: tla.mutex_lock %[[DST]][<vector>] : !tla.mutex
// CHECK-NEXT: "tla.vec.func"() ({
// CHECK-NEXT: %[[VALUE:.*]] = tla.load
// CHECK-NEXT: %[[SIGNED:.*]] = tla.load
// CHECK-NEXT: %[[INDICES:.*]] = tla.bitcast %[[SIGNED]]
// CHECK-NEXT: tla.scatter %{{.*}}, %[[VALUE]], %[[INDICES]]
// CHECK-NEXT: tla.scatter %{{.*}}, %[[VALUE]], %[[INDICES]]
// CHECK: tla.mutex_unlock %[[DST]][<vector>] : !tla.mutex
// CHECK-NEXT: tla.mutex_unlock %[[IDX]][<vector>] : !tla.mutex
// CHECK-NEXT: tla.mutex_unlock %[[SRC]][<vector>] : !tla.mutex
// CHECK-NOT: tla.mutex_lock
// CHECK: return
