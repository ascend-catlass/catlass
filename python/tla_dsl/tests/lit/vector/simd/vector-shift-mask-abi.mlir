// RUN: %tla_compile %s --mlir-print-ir-after=tla-vector-region-simd -o %t 2>&1 | %filecheck %s
// RUN: sed '/%counts =/!s/!tla.vector<32xi64>/!tla.vector<32xui64>/g; /tla.shift_\(left\|right\) /s/, !tla.vector<32xui64> mask/, !tla.vector<32xi64> mask/' %s | %tla_compile - --mlir-print-ir-after=tla-vector-region-simd -o %t 2>&1 | %filecheck %s --check-prefix=UNSIGNED

!bits = !tla.tensor<!tla.layout<!tla.shape<4>, !tla.stride<1>, !tla.shape<4>, RowMajor>, !tla.coord<0>, !tla.ptr<i8, ub, 1>>
!mask = !tla.mask<32>

module {
  func.func @shift_mask_sources(%bits_mem: memref<4xi8, #hivm.address_space<ub>>, %address: i64, %count: index, %choose: i1) {
    %zero = arith.constant 0 : index
    %one = arith.constant 1 : index
    %four = arith.constant 4 : index
    %bits = tla.tensor_desc %bits_mem shape [%one, %four, %one, %one] stride [%four, %one, %one, %one] origin_shape [%one, %four] coord [%zero, %zero] : memref<4xi8, #hivm.address_space<ub>> -> !bits
    %ptr = tla.inttoptr %address : i64 -> !tla.ptr<i8, ub, 4>
    "tla.vec.func"() ({
      %data = tla.call_extern @data(%ptr) : (!tla.ptr<i8, ub, 4>) -> !tla.vector<32xi64>
      %counts = tla.call_extern @counts(%ptr) : (!tla.ptr<i8, ub, 4>) -> !tla.vector<32xi64>
      %loaded = tla.load %bits : !bits -> !mask
      %tail, %remaining = tla.update_mask %count, i64 : !mask, index
      %pattern = "tla.create_mask"() {dtype = i64, pattern = "M3"} : () -> !mask
      %selected = "tla.where"(%pattern, %loaded, %tail) : (!mask, !mask, !mask) -> !mask
      %carried = scf.for %i = %zero to %count step %one iter_args(%m = %selected) -> (!mask) {
        %next = scf.if %choose -> (!mask) {
          scf.yield %m : !mask
        } else {
          scf.yield %tail : !mask
        }
        scf.yield %next : !mask
      }
      %scalar = arith.constant 4 : i16
      %a = tla.shift_left %data, %counts mask %loaded : !tla.vector<32xi64>, !tla.vector<32xi64> mask !mask -> !tla.vector<32xi64>
      %b = tla.shift_right %a, %counts mask %tail : !tla.vector<32xi64>, !tla.vector<32xi64> mask !mask -> !tla.vector<32xi64>
      %c = tla.shift_lefts %b, %scalar mask %selected : !tla.vector<32xi64>, i16 mask !mask -> !tla.vector<32xi64>
      %d = tla.shift_rights %c, %scalar mask %carried : !tla.vector<32xi64>, i16 mask !mask -> !tla.vector<32xi64>
      tla.call_extern @sink(%d) : (!tla.vector<32xi64>) -> ()
    }) {mode = "simd"} : () -> ()
    return
  }
}

// CHECK-DAG: func.func private @shift_left_int64_t(vector<32xi64>, vector<32xi64>, vector<256xi1>) -> vector<32xi64>
// CHECK-DAG: func.func private @shift_right_int64_t(vector<32xi64>, vector<32xi64>, vector<256xi1>) -> vector<32xi64>
// CHECK-DAG: func.func private @shift_lefts_int64_t(vector<32xi64>, i16, vector<256xi1>) -> vector<32xi64>
// CHECK-DAG: func.func private @shift_rights_int64_t(vector<32xi64>, i16, vector<256xi1>) -> vector<32xi64>
// CHECK-LABEL: func.func private @vector_region_
// CHECK: builtin.unrealized_conversion_cast {{.*}} : vector<32xi1> to vector<256xi1>
// CHECK: scf.for {{.*}} -> (vector<256xi1>)
// CHECK: scf.if {{.*}} -> (vector<256xi1>)
// CHECK: call @shift_left_int64_t({{.*}}) : (vector<32xi64>, vector<32xi64>, vector<256xi1>) -> vector<32xi64>
// CHECK: call @shift_right_int64_t({{.*}}) : (vector<32xi64>, vector<32xi64>, vector<256xi1>) -> vector<32xi64>
// CHECK: call @shift_lefts_int64_t({{.*}}) : (vector<32xi64>, i16, vector<256xi1>) -> vector<32xi64>
// CHECK: call @shift_rights_int64_t({{.*}}) : (vector<32xi64>, i16, vector<256xi1>) -> vector<32xi64>

// UNSIGNED-DAG: func.func private @shift_left_uint64_t(vector<32xi64>, vector<32xi64>, vector<256xi1>) -> vector<32xi64>
// UNSIGNED-DAG: func.func private @shift_right_uint64_t(vector<32xi64>, vector<32xi64>, vector<256xi1>) -> vector<32xi64>
// UNSIGNED-DAG: func.func private @shift_lefts_uint64_t(vector<32xi64>, i16, vector<256xi1>) -> vector<32xi64>
// UNSIGNED-DAG: func.func private @shift_rights_uint64_t(vector<32xi64>, i16, vector<256xi1>) -> vector<32xi64>
// UNSIGNED-LABEL: func.func private @vector_region_
// UNSIGNED: call @shift_left_uint64_t({{.*}}) : (vector<32xi64>, vector<32xi64>, vector<256xi1>) -> vector<32xi64>
// UNSIGNED: call @shift_right_uint64_t({{.*}}) : (vector<32xi64>, vector<32xi64>, vector<256xi1>) -> vector<32xi64>
// UNSIGNED: call @shift_lefts_uint64_t({{.*}}) : (vector<32xi64>, i16, vector<256xi1>) -> vector<32xi64>
// UNSIGNED: call @shift_rights_uint64_t({{.*}}) : (vector<32xi64>, i16, vector<256xi1>) -> vector<32xi64>
