// RUN: %tla_compile %s --mlir-print-ir-after=tla-unalign-ureg -o %t 2>&1 | %filecheck %s --check-prefix=STATE
// RUN: %tla_compile %s -o - | %filecheck %s --check-prefix=LOWER

!tile = !tla.tensor<!tla.layout<!tla.shape<64>, !tla.stride<1>, !tla.shape<64>, RowMajor>, !tla.coord<0>, !tla.ptr<f32, ub, 4>>
!vec = !tla.vector<64xf32>
!ptr = !tla.ptr<f32, ub, 4>

module {
  func.func @last_base(
      %source: memref<64xf32, #hivm.address_space<ub>>,
      %dst: memref<64xf32, #hivm.address_space<ub>>,
      %other: memref<64xf32, #hivm.address_space<ub>>, %trips: i32) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c64 = arith.constant 64 : index
    %src = tla.tensor_desc %source shape [%c1, %c64, %c1, %c1] stride [%c64, %c1, %c1, %c1] origin_shape [%c1, %c64] coord [%c0, %c0] : memref<64xf32, #hivm.address_space<ub>> -> !tile
    %a = tla.tensor_desc %dst shape [%c1, %c64, %c1, %c1] stride [%c64, %c1, %c1, %c1] origin_shape [%c1, %c64] coord [%c0, %c0] : memref<64xf32, #hivm.address_space<ub>> -> !tile
    %b = tla.tensor_desc %other shape [%c1, %c64, %c1, %c1] stride [%c64, %c1, %c1, %c1] origin_shape [%c1, %c64] coord [%c0, %c0] : memref<64xf32, #hivm.address_space<ub>> -> !tile
    "tla.vec.func"() ({
      %lo = arith.constant 0 : i32
      %step = arith.constant 1 : i32
      %values = tla.load %src : !tile -> !vec
      %keep = "tla.create_mask"() {pattern = "VL8", dtype = f32} : () -> !tla.mask<64>
      scf.for %i = %lo to %trips step %step : i32 {
        %packed = tla.squeeze %values, %keep {store_reg = true} : !vec, !tla.mask<64> -> !vec
        %first = arith.cmpi eq, %i, %lo : i32
        scf.if %first {
          %ptr, %state = tla.store_unalign_update %a, %packed post_update_stride %c0 {compressed = true} : !tile, !vec post_update_stride index -> !ptr, !tla.ureg
        } else {
          %ptr, %state = tla.store_unalign_update %b, %packed post_update_stride %c0 {compressed = true} : !tile, !vec post_update_stride index -> !ptr, !tla.ureg
        }
      }
    }) {mode = "simd"} : () -> ()
    return
  }
}

// STATE: [[A:%[^ ]+]] = tla.tensor_desc %arg1
// STATE: [[B:%[^ ]+]] = tla.tensor_desc %arg2
// STATE: tla.clear_squeeze_state
// STATE: [[EMPTY:%[^ ]+]] = arith.constant -1 : i64
// STATE: [[LOOP:%[^ ]+]]:2 = scf.for {{.*}} -> (!tla.ureg, i64)
// STATE: [[CHOSEN:%[^ ]+]]:2 = scf.if {{.*}} -> (!tla.ureg, i64)
// STATE: [[AP:%[^ ]+]], [[AS:%[^ ]+]] = tla.store_unalign_update [[A]],
// STATE-NEXT: [[AB:%[^ ]+]] = tla.ptrtoint [[AP]]
// STATE-NEXT: scf.yield [[AS]], [[AB]]
// STATE: [[BP:%[^ ]+]], [[BS:%[^ ]+]] = tla.store_unalign_update [[B]],
// STATE-NEXT: [[BB:%[^ ]+]] = tla.ptrtoint [[BP]]
// STATE-NEXT: scf.yield [[BS]], [[BB]]
// STATE: scf.yield [[CHOSEN]]#0, [[CHOSEN]]#1
// STATE: [[HAS:%[^ ]+]] = arith.cmpi ne, [[LOOP]]#1, [[EMPTY]]
// STATE-NEXT: scf.if [[HAS]]
// STATE-NEXT: tla.store_unalign_post [[LOOP]]#0, [[LOOP]]#1 {compressed = true}

// LOWER: call @squeeze_store_float(
// LOWER: call @squeeze_store_float(
// LOWER: call @squeeze_store_post_uint8_t(
