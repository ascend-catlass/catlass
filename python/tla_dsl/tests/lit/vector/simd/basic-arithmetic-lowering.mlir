// RUN: %tla_compile %s --mlir-print-ir-after=tla-vector-region-simd -o %t > %t.ir 2>&1
// RUN: %filecheck %s < %t.ir

!tile = !tla.tensor<!tla.layout<!tla.shape<32>, !tla.stride<1>, !tla.shape<32>, RowMajor>, !tla.coord<0>, !tla.ptr<i64, ub, 32>>
!out = !tla.tensor<!tla.layout<!tla.shape<64>, !tla.stride<1>, !tla.shape<64>, RowMajor>, !tla.coord<0>, !tla.ptr<i32, ub, 32>>
!i64reg = !tla.vector<32xi64>
!u64reg = !tla.vector<32xui64>
!u16reg = !tla.vector<128xui16>
!b64mask = !tla.mask<32>

module {
  func.func @signed(%src_memref: memref<32xi64, #hivm.address_space<ub>>,
                    %dst_memref: memref<64xi32, #hivm.address_space<ub>>, %scalar: i64) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c32 = arith.constant 32 : index
    %c64 = arith.constant 64 : index
    %src = tla.tensor_desc %src_memref shape [%c1, %c32, %c1, %c1] stride [%c32, %c1, %c1, %c1] origin_shape [%c1, %c32] coord [%c0, %c0] : memref<32xi64, #hivm.address_space<ub>> -> !tile
    %dst = tla.tensor_desc %dst_memref shape [%c1, %c64, %c1, %c1] stride [%c64, %c1, %c1, %c1] origin_shape [%c1, %c64] coord [%c0, %c0] : memref<64xi32, #hivm.address_space<ub>> -> !out
    "tla.vec.func"() ({
      %x = tla.load %src : !tile -> !i64reg
      %a = tla.abs %x : !i64reg -> !i64reg
      %b = tla.adds %a, %scalar : !i64reg, i64 -> !i64reg
      %mask = "tla.create_mask"() {dtype = i64, pattern = "VL8"} : () -> !b64mask
      %c = tla.abs %x mask %mask : !i64reg mask !b64mask -> !i64reg
      %d = tla.min %b, %c mask %mask : !i64reg, !i64reg mask !b64mask -> !i64reg
      %words = tla.bitcast %d : !i64reg -> !tla.vector<64xi32>
      tla.store %dst, %words : !out, !tla.vector<64xi32>
    }) {mode = "simd"} : () -> ()
    return
  }

  func.func @unsigned(%address: i64, %wide: ui64, %narrow: ui16) {
    %ptr = tla.inttoptr %address : i64 -> !tla.ptr<i8, ub, 4>
    "tla.vec.func"() ({
      %x = tla.call_extern @source_u64(%ptr) : (!tla.ptr<i8, ub, 4>) -> !u64reg
      %y = tla.call_extern @source_u16(%ptr) : (!tla.ptr<i8, ub, 4>) -> !u16reg
      %a = tla.bitwise_not %x : !u64reg -> !u64reg
      %mask = "tla.create_mask"() {dtype = ui64, pattern = "VL8"} : () -> !b64mask
      %b = tla.mins %a, %wide mask %mask : !u64reg, ui64 mask !b64mask -> !u64reg
      %c = tla.adds %y, %narrow : !u16reg, ui16 -> !u16reg
      tla.call_extern @sink_unsigned(%b, %c) : (!u64reg, !u16reg) -> ()
    }) {mode = "simd"} : () -> ()
    return
  }
}

// CHECK-LABEL: func.func private @vector_region_0
// CHECK: %[[X:.*]] = vector.bitcast {{.*}} : vector<64xi32> to vector<32xi64>
// CHECK: %[[IMPLICIT:.*]] = "hivm_regbaseintrins.intr.hivm.punpack"
// CHECK: call @tla_abs_i64(%[[X]], %[[IMPLICIT]]) : (vector<32xi64>, vector<256xi1>) -> vector<32xi64>
// CHECK: call @tla_adds_i64({{.*}}) : (vector<32xi64>, i64, vector<256xi1>) -> vector<32xi64>
// CHECK: %[[PATTERN:.*]] = ave.hir.pge <VL8>
// CHECK: %[[EXPLICIT:.*]] = "hivm_regbaseintrins.intr.hivm.punpack"(%[[PATTERN]],
// CHECK-NOT: hivm.punpack
// CHECK: call @tla_abs_zeroing_i64(%[[X]], %[[EXPLICIT]]) : (vector<32xi64>, vector<256xi1>) -> vector<32xi64>
// CHECK-NOT: hivm.punpack
// CHECK: call @tla_min_i64({{.*}}, %[[EXPLICIT]]) : (vector<32xi64>, vector<32xi64>, vector<256xi1>) -> vector<32xi64>

// CHECK-LABEL: func.func private @vector_region_1
// CHECK: %[[UMASK:.*]] = "hivm_regbaseintrins.intr.hivm.punpack"
// CHECK: call @tla_bitwise_not_u64({{.*}}, %[[UMASK]]) : (vector<32xi64>, vector<256xi1>) -> vector<32xi64>
// CHECK: %[[UPATTERN:.*]] = ave.hir.pge <VL8>
// CHECK: %[[UEXPLICIT:.*]] = "hivm_regbaseintrins.intr.hivm.punpack"(%[[UPATTERN]],
// CHECK-NOT: hivm.punpack
// CHECK: call @tla_mins_u64({{.*}}, %[[UEXPLICIT]]) : (vector<32xi64>, i64, vector<256xi1>) -> vector<32xi64>
// CHECK: %[[CARRIER:.*]] = builtin.unrealized_conversion_cast {{.*}} : ui16 to i16
// CHECK: ave.hir.vadds {{.*}}%[[CARRIER]]
