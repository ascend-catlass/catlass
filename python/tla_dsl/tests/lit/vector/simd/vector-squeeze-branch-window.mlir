// RUN: %tla_compile %s --mlir-print-ir-after=tla-unalign-ureg -o %t 2>&1 | %filecheck %s --check-prefix=STATE
// RUN: %tla_compile %s --mlir-print-ir-after=tla-vector-region-simd -o %t 2>&1 | %filecheck %s --check-prefix=LOWER

!tile = !tla.tensor<!tla.layout<!tla.shape<64>, !tla.stride<1>, !tla.shape<64>, RowMajor>, !tla.coord<0>, !tla.ptr<f32, ub, 256>>
!storage = !tla.tensor<!tla.layout<!tla.shape<128>, !tla.stride<1>, !tla.shape<128>, RowMajor>, !tla.coord<0>, !tla.ptr<f32, ub, 256>>
!vec = !tla.vector<64xf32>
!ptr = !tla.ptr<f32, ub, 256>

module {
  func.func @branch_local_window(%enabled: i1, %trips: index) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c64 = arith.constant 64 : index
    %c128 = arith.constant 128 : index
    %src_ptr = "tla.alloc_ptr"() {size_bytes = 256 : i64} : () -> !ptr
    %dst_ptr = "tla.alloc_ptr"() {size_bytes = 512 : i64} : () -> !ptr
    %src = tla.tensor_desc %src_ptr shape [%c1, %c64, %c1, %c1] stride [%c64, %c1, %c1, %c1] origin_shape [%c1, %c64] coord [%c0, %c0] : !ptr -> !tile
    %dst = tla.tensor_desc %dst_ptr shape [%c1, %c128, %c1, %c1] stride [%c128, %c1, %c1, %c1] origin_shape [%c1, %c128] coord [%c0, %c0] : !ptr -> !storage
    "tla.vec.func"() ({
      scf.if %enabled {
        %shape = tla.make_shape -> !tla.shape<64>
        %coord = tla.make_coord -> !tla.coord<0>
        %window = tla.tile_view %dst, %shape, %coord : !storage, !tla.shape<64>, !tla.coord<0> -> !tile
        %values = tla.load %src : !tile -> !vec
        %keep = "tla.create_mask"() {pattern = "VL1", dtype = f32} : () -> !tla.mask<64>
        scf.for %i = %c0 to %trips step %c1 {
          %packed = tla.squeeze %values, %keep {store_reg = true} : !vec, !tla.mask<64> -> !vec
          %next, %state = tla.store_unalign_update %window, %packed post_update_stride %c0 {compressed = true} : !tile, !vec post_update_stride index -> !ptr, !tla.ureg
        }
      }
    }) {mode = "simd"} : () -> ()
    return
  }

  func.func @loop_branch_local_window(%trips: i32, %limit: i32, %offset: index) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c64 = arith.constant 64 : index
    %zero = arith.constant 0 : i32
    %one = arith.constant 1 : i32
    %dst_ptr = "tla.alloc_ptr"() {size_bytes = 512 : i64} : () -> !ptr
    %count_ptr = "tla.alloc_ptr"() {size_bytes = 256 : i64} : () -> !ptr
    %count_out = tla.tensor_desc %count_ptr shape [%c1, %c64, %c1, %c1] stride [%c64, %c1, %c1, %c1] origin_shape [%c1, %c64] coord [%c0, %c0] : !ptr -> !tile
    "tla.vec.func"() ({
      %value = arith.constant 11.0 : f32
      %values = tla.full %value : f32 -> !vec
      %keep = "tla.create_mask"() {pattern = "VL1", dtype = f32} : () -> !tla.mask<64>
      %count = scf.for %i = %zero to %trips step %one iter_args(%previous_count = %zero) -> (i32) : i32 {
        %append = arith.cmpi slt, %i, %limit : i32
        %next_count = scf.if %append -> (i32) {
          %ptr = tla.ptr_add %dst_ptr, %offset : !ptr, index -> !ptr
          %window = tla.tensor_desc %ptr shape [%c1, %c64, %c1, %c1] stride [%c64, %c1, %c1, %c1] origin_shape [%c1, %c64] coord [%c0, %c0] : !ptr -> !tile
          %packed = tla.squeeze %values, %keep {store_reg = true} : !vec, !tla.mask<64> -> !vec
          %next, %state = tla.store_unalign_update %window, %packed post_update_stride %c0 {compressed = true} : !tile, !vec post_update_stride index -> !ptr, !tla.ureg
          %incremented = arith.addi %previous_count, %one : i32
          scf.yield %incremented : i32
        } else {
          scf.yield %previous_count : i32
        }
        scf.yield %next_count : i32
      }
      %count_vector = tla.full %count : i32 -> !tla.vector<64xi32>
      %summary = tla.bitcast %count_vector : !tla.vector<64xi32> -> !vec
      tla.store %count_out, %summary : !tile, !vec
    }) {mode = "simd"} : () -> ()
    return
  }
}

// STATE-LABEL: func.func @branch_local_window
// STATE: "tla.vec.func"
// STATE-NEXT: tla.clear_squeeze_state
// STATE-NEXT: {{.*}} = tla.create_ureg
// STATE: [[BRANCH:%[a-zA-Z0-9_]+]]:2 = scf.if
// STATE: [[WINDOW:%[a-zA-Z0-9_]+]] = tla.tile_view
// STATE: scf.for
// STATE: tla.store_unalign_update [[WINDOW]],
// STATE: tla.ptrtoint
// STATE: scf.yield
// STATE: arith.cmpi ne, [[BRANCH]]#1,
// STATE-NEXT: scf.if
// STATE-NEXT: tla.store_unalign_post [[BRANCH]]#0, [[BRANCH]]#1 {compressed = true}

// STATE-LABEL: func.func @loop_branch_local_window
// STATE: "tla.vec.func"
// STATE-NEXT: tla.clear_squeeze_state
// STATE-NEXT: [[EMPTY:%[^ ]+]] = tla.create_ureg
// STATE-NEXT: [[NO_APPEND:%[^ ]+]] = arith.constant -1 : i64
// STATE: [[LOOP:%[^ ]+]]:3 = scf.for
// STATE-SAME: iter_args([[COUNT:%[^ ]+]] = {{%[^, ]+}}, [[UREG:%[^ ]+]] = [[EMPTY]], [[ADDR:%[^ ]+]] = [[NO_APPEND]]) -> (i32, !tla.ureg, i64)
// STATE: [[APPEND:%[^ ]+]]:3 = scf.if
// STATE: [[NEXT:%[^, ]+]], [[TAIL:%[^ ]+]] = tla.store_unalign_update
// STATE-NEXT: [[DEST:%[^ ]+]] = tla.ptrtoint [[NEXT]]
// STATE: scf.yield {{%[^, ]+}}, [[TAIL]], [[DEST]] : i32, !tla.ureg, i64
// STATE: } else {
// STATE-NEXT: scf.yield [[COUNT]], [[UREG]], [[ADDR]] : i32, !tla.ureg, i64
// STATE: scf.yield [[APPEND]]#0, [[APPEND]]#1, [[APPEND]]#2 : i32, !tla.ureg, i64
// STATE: [[HAS_APPEND:%[^ ]+]] = arith.cmpi ne, [[LOOP]]#2, [[NO_APPEND]]
// STATE-NEXT: scf.if [[HAS_APPEND]]
// STATE-NEXT: tla.store_unalign_post [[LOOP]]#1, [[LOOP]]#2 {compressed = true}

// LOWER-LABEL: func.func private @vector_region_{{[0-9]+}}(
// LOWER: call @squeeze_stream_clear
// LOWER: scf.if
// LOWER: scf.while
// LOWER: call @squeeze_store_float(
// LOWER: call @squeeze_store_post_uint8_t(
// LOWER-LABEL: func.func private @vector_region_{{[0-9]+}}(
// LOWER: call @squeeze_stream_clear
// LOWER: scf.while
// LOWER: call @squeeze_store_float(
// LOWER: call @squeeze_store_post_uint8_t(
