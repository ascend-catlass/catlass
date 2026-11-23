// RUN: %tla_compile %s --mlir-print-ir-after=tla-vector-region-simd -o %t 2>&1 | %filecheck %s --check-prefix=UNSIGNED
// RUN: sed 's/64xui32/128xui16/g;s/64xi32/128xi16/g;s/mask<64>/mask<128>/g;s/dtype = ui32/dtype = ui16/' %s | %tla_compile - --mlir-print-ir-after=tla-vector-region-simd -o %t 2>&1 | %filecheck %s --check-prefix=UNSIGNED
// RUN: sed 's/64xui32/256xui8/g;s/64xi32/256xi8/g;s/mask<64>/mask<256>/g;s/dtype = ui32/dtype = ui8/' %s | %tla_compile - --mlir-print-ir-after=tla-vector-region-simd -o %t 2>&1 | %filecheck %s --check-prefix=UNSIGNED
// RUN: sed 's/64xui32/32xui64/g;s/64xi32/32xi64/g;s/mask<64>/mask<32>/g;s/dtype = ui32/dtype = ui64/' %s | %tla_compile - --mlir-print-ir-after=tla-vector-region-simd -o %t 2>&1 | %filecheck %s --check-prefix=BC --implicit-check-not='hivm.vshr.s.x'
// RUN: sed 's/64xui32/32xi64/g;s/64xi32/32xi64/g;s/mask<64>/mask<32>/g;s/dtype = ui32/dtype = i64/' %s | %tla_compile - --mlir-print-ir-after=tla-vector-region-simd -o %t 2>&1 | %filecheck %s --check-prefix=SIGNED64

module {
  func.func @shift(%address: i64) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>} {
    %ptr = tla.inttoptr %address : i64 -> !tla.ptr<i8, ub, 4>
    "tla.vec.func"() ({
      %a = tla.call_extern @data(%ptr) : (!tla.ptr<i8, ub, 4>) -> !tla.vector<64xui32>
      %b = tla.call_extern @amount(%ptr) : (!tla.ptr<i8, ub, 4>) -> !tla.vector<64xi32>
      %mask = "tla.create_mask"() {dtype = ui32, pattern = "ALL"} : () -> !tla.mask<64>
      %scalar = arith.constant 4 : i16
      %r0 = tla.shift_left %a, %b mask %mask : !tla.vector<64xui32>, !tla.vector<64xi32> mask !tla.mask<64> -> !tla.vector<64xui32>
      %r1 = tla.shift_right %r0, %b mask %mask : !tla.vector<64xui32>, !tla.vector<64xi32> mask !tla.mask<64> -> !tla.vector<64xui32>
      %r2 = tla.shift_lefts %r1, %scalar mask %mask : !tla.vector<64xui32>, i16 mask !tla.mask<64> -> !tla.vector<64xui32>
      %r3 = tla.shift_rights %r2, %scalar mask %mask : !tla.vector<64xui32>, i16 mask !tla.mask<64> -> !tla.vector<64xui32>
      tla.call_extern @sink(%r3) : (!tla.vector<64xui32>) -> ()
    }) : () -> ()
    return
  }
}

// UNSIGNED-LABEL: func.func private @vector_region_
// UNSIGNED: hivm.vshl.u.x
// UNSIGNED: hivm.vshr.u.x
// UNSIGNED: hivm.vshls.u.x
// UNSIGNED: hivm.vshrs.u.x

// BC-DAG: func.func private @shift_left_uint64_t(vector<32xi64>, vector<32xi64>, vector<256xi1>) -> vector<32xi64>
// BC-DAG: func.func private @shift_right_uint64_t(vector<32xi64>, vector<32xi64>, vector<256xi1>) -> vector<32xi64>
// BC-DAG: func.func private @shift_lefts_uint64_t(vector<32xi64>, i16, vector<256xi1>) -> vector<32xi64>
// BC-DAG: func.func private @shift_rights_uint64_t(vector<32xi64>, i16, vector<256xi1>) -> vector<32xi64>
// BC-LABEL: func.func private @vector_region_
// BC: %[[UCOUNT:[[:alnum:]_]+]] = call @amount(
// BC: hivm.punpack
// BC: call @shift_left_uint64_t({{.*}}, %[[UCOUNT]], {{.*}}) : (vector<32xi64>, vector<32xi64>, vector<256xi1>) -> vector<32xi64>
// BC: call @shift_right_uint64_t({{.*}}, %[[UCOUNT]], {{.*}}) : (vector<32xi64>, vector<32xi64>, vector<256xi1>) -> vector<32xi64>
// BC: call @shift_lefts_uint64_t({{.*}}) : (vector<32xi64>, i16, vector<256xi1>) -> vector<32xi64>
// BC: call @shift_rights_uint64_t({{.*}}) : (vector<32xi64>, i16, vector<256xi1>) -> vector<32xi64>

// SIGNED64-DAG: func.func private @shift_left_int64_t(vector<32xi64>, vector<32xi64>, vector<256xi1>) -> vector<32xi64>
// SIGNED64-DAG: func.func private @shift_right_int64_t(vector<32xi64>, vector<32xi64>, vector<256xi1>) -> vector<32xi64>
// SIGNED64-DAG: func.func private @shift_lefts_int64_t(vector<32xi64>, i16, vector<256xi1>) -> vector<32xi64>
// SIGNED64-DAG: func.func private @shift_rights_int64_t(vector<32xi64>, i16, vector<256xi1>) -> vector<32xi64>
// SIGNED64-LABEL: func.func private @vector_region_
// SIGNED64: %[[SCOUNT:[[:alnum:]_]+]] = call @amount(
// SIGNED64: call @shift_left_int64_t({{.*}}, %[[SCOUNT]], {{.*}}) : (vector<32xi64>, vector<32xi64>, vector<256xi1>) -> vector<32xi64>
// SIGNED64: call @shift_right_int64_t({{.*}}, %[[SCOUNT]], {{.*}}) : (vector<32xi64>, vector<32xi64>, vector<256xi1>) -> vector<32xi64>
// SIGNED64: call @shift_lefts_int64_t({{.*}}) : (vector<32xi64>, i16, vector<256xi1>) -> vector<32xi64>
// SIGNED64: call @shift_rights_int64_t({{.*}}) : (vector<32xi64>, i16, vector<256xi1>) -> vector<32xi64>
