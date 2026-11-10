// RUN: %tla_compile %s --mlir-print-ir-after=tla-vector-region-simd -o %t 2>&1 | %filecheck %s

// Each full register type can be returned, passed to another extern, and
// converted to a different register type by an extern. Unsigned ABI vectors
// become signless, while source declarations retain their signedness.

module {
  func.func @register_types(%addr: i64) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>} {
    %p = tla.inttoptr %addr : i64 -> !tla.ptr<i8, ub, 4>
    "tla.vec.func"() ({
      %a_f16 = tla.call_extern @read_f16(%p) : (!tla.ptr<i8, ub, 4>) -> !tla.vector<128xf16>
      %b_f16 = tla.call_extern @step_f16(%a_f16, %p) : (!tla.vector<128xf16>, !tla.ptr<i8, ub, 4>) -> !tla.vector<128xf16>
      %c_f16 = tla.call_extern @to_bytes_f16(%b_f16) : (!tla.vector<128xf16>) -> !tla.vector<256xui8>
      tla.call_extern @sink(%c_f16, %p) : (!tla.vector<256xui8>, !tla.ptr<i8, ub, 4>) -> ()
      %a_bf16 = tla.call_extern @read_bf16(%p) : (!tla.ptr<i8, ub, 4>) -> !tla.vector<128xbf16>
      %b_bf16 = tla.call_extern @step_bf16(%a_bf16, %p) : (!tla.vector<128xbf16>, !tla.ptr<i8, ub, 4>) -> !tla.vector<128xbf16>
      %c_bf16 = tla.call_extern @to_bytes_bf16(%b_bf16) : (!tla.vector<128xbf16>) -> !tla.vector<256xui8>
      tla.call_extern @sink(%c_bf16, %p) : (!tla.vector<256xui8>, !tla.ptr<i8, ub, 4>) -> ()
      %a_f32 = tla.call_extern @read_f32(%p) : (!tla.ptr<i8, ub, 4>) -> !tla.vector<64xf32>
      %b_f32 = tla.call_extern @step_f32(%a_f32, %p) : (!tla.vector<64xf32>, !tla.ptr<i8, ub, 4>) -> !tla.vector<64xf32>
      %c_f32 = tla.call_extern @to_bytes_f32(%b_f32) : (!tla.vector<64xf32>) -> !tla.vector<256xui8>
      tla.call_extern @sink(%c_f32, %p) : (!tla.vector<256xui8>, !tla.ptr<i8, ub, 4>) -> ()
      %a_i8 = tla.call_extern @read_i8(%p) : (!tla.ptr<i8, ub, 4>) -> !tla.vector<256xi8>
      %b_i8 = tla.call_extern @step_i8(%a_i8, %p) : (!tla.vector<256xi8>, !tla.ptr<i8, ub, 4>) -> !tla.vector<256xi8>
      %c_i8 = tla.call_extern @to_bytes_i8(%b_i8) : (!tla.vector<256xi8>) -> !tla.vector<256xui8>
      tla.call_extern @sink(%c_i8, %p) : (!tla.vector<256xui8>, !tla.ptr<i8, ub, 4>) -> ()
      %a_ui8 = tla.call_extern @read_ui8(%p) : (!tla.ptr<i8, ub, 4>) -> !tla.vector<256xui8>
      %b_ui8 = tla.call_extern @step_ui8(%a_ui8, %p) : (!tla.vector<256xui8>, !tla.ptr<i8, ub, 4>) -> !tla.vector<256xui8>
      %c_ui8 = tla.call_extern @to_bytes_ui8(%b_ui8) : (!tla.vector<256xui8>) -> !tla.vector<256xui8>
      tla.call_extern @sink(%c_ui8, %p) : (!tla.vector<256xui8>, !tla.ptr<i8, ub, 4>) -> ()
      %a_i16 = tla.call_extern @read_i16(%p) : (!tla.ptr<i8, ub, 4>) -> !tla.vector<128xi16>
      %b_i16 = tla.call_extern @step_i16(%a_i16, %p) : (!tla.vector<128xi16>, !tla.ptr<i8, ub, 4>) -> !tla.vector<128xi16>
      %c_i16 = tla.call_extern @to_bytes_i16(%b_i16) : (!tla.vector<128xi16>) -> !tla.vector<256xui8>
      tla.call_extern @sink(%c_i16, %p) : (!tla.vector<256xui8>, !tla.ptr<i8, ub, 4>) -> ()
      %a_ui16 = tla.call_extern @read_ui16(%p) : (!tla.ptr<i8, ub, 4>) -> !tla.vector<128xui16>
      %b_ui16 = tla.call_extern @step_ui16(%a_ui16, %p) : (!tla.vector<128xui16>, !tla.ptr<i8, ub, 4>) -> !tla.vector<128xui16>
      %c_ui16 = tla.call_extern @to_bytes_ui16(%b_ui16) : (!tla.vector<128xui16>) -> !tla.vector<256xui8>
      tla.call_extern @sink(%c_ui16, %p) : (!tla.vector<256xui8>, !tla.ptr<i8, ub, 4>) -> ()
      %a_i32 = tla.call_extern @read_i32(%p) : (!tla.ptr<i8, ub, 4>) -> !tla.vector<64xi32>
      %b_i32 = tla.call_extern @step_i32(%a_i32, %p) : (!tla.vector<64xi32>, !tla.ptr<i8, ub, 4>) -> !tla.vector<64xi32>
      %c_i32 = tla.call_extern @to_bytes_i32(%b_i32) : (!tla.vector<64xi32>) -> !tla.vector<256xui8>
      tla.call_extern @sink(%c_i32, %p) : (!tla.vector<256xui8>, !tla.ptr<i8, ub, 4>) -> ()
      %a_ui32 = tla.call_extern @read_ui32(%p) : (!tla.ptr<i8, ub, 4>) -> !tla.vector<64xui32>
      %b_ui32 = tla.call_extern @step_ui32(%a_ui32, %p) : (!tla.vector<64xui32>, !tla.ptr<i8, ub, 4>) -> !tla.vector<64xui32>
      %c_ui32 = tla.call_extern @to_bytes_ui32(%b_ui32) : (!tla.vector<64xui32>) -> !tla.vector<256xui8>
      tla.call_extern @sink(%c_ui32, %p) : (!tla.vector<256xui8>, !tla.ptr<i8, ub, 4>) -> ()
      %a_i64 = tla.call_extern @read_i64(%p) : (!tla.ptr<i8, ub, 4>) -> !tla.vector<32xi64>
      %b_i64 = tla.call_extern @step_i64(%a_i64, %p) : (!tla.vector<32xi64>, !tla.ptr<i8, ub, 4>) -> !tla.vector<32xi64>
      %c_i64 = tla.call_extern @to_bytes_i64(%b_i64) : (!tla.vector<32xi64>) -> !tla.vector<256xui8>
      tla.call_extern @sink(%c_i64, %p) : (!tla.vector<256xui8>, !tla.ptr<i8, ub, 4>) -> ()
      %a_ui64 = tla.call_extern @read_ui64(%p) : (!tla.ptr<i8, ub, 4>) -> !tla.vector<32xui64>
      %b_ui64 = tla.call_extern @step_ui64(%a_ui64, %p) : (!tla.vector<32xui64>, !tla.ptr<i8, ub, 4>) -> !tla.vector<32xui64>
      %c_ui64 = tla.call_extern @to_bytes_ui64(%b_ui64) : (!tla.vector<32xui64>) -> !tla.vector<256xui8>
      tla.call_extern @sink(%c_ui64, %p) : (!tla.vector<256xui8>, !tla.ptr<i8, ub, 4>) -> ()
    }) : () -> ()
    return
  }
}

// CHECK-DAG: func.func private @step_f16(vector<128xf16>, i64) -> vector<128xf16>
// CHECK-DAG: func.func private @step_bf16(vector<128xbf16>, i64) -> vector<128xbf16>
// CHECK-DAG: func.func private @step_f32(vector<64xf32>, i64) -> vector<64xf32>
// CHECK-DAG: func.func private @step_i8(vector<256xi8>, i64) -> vector<256xi8>
// CHECK-DAG: func.func private @step_ui8(vector<256xi8>, i64) -> vector<256xi8>
// CHECK-DAG: func.func private @step_i16(vector<128xi16>, i64) -> vector<128xi16>
// CHECK-DAG: func.func private @step_ui16(vector<128xi16>, i64) -> vector<128xi16>
// CHECK-DAG: func.func private @step_i32(vector<64xi32>, i64) -> vector<64xi32>
// CHECK-DAG: func.func private @step_ui32(vector<64xi32>, i64) -> vector<64xi32>
// CHECK-DAG: func.func private @step_i64(vector<32xi64>, i64) -> vector<32xi64>
// CHECK-DAG: func.func private @step_ui64(vector<32xi64>, i64) -> vector<32xi64>
// CHECK-LABEL: func.func private @vector_region_0
// CHECK: %[[A_f16:[^ ]+]] = call @read_f16({{.*}}) : (i64) -> vector<128xf16>
// CHECK: %[[B_f16:[^ ]+]] = call @step_f16(%[[A_f16]], {{.*}}) : (vector<128xf16>, i64) -> vector<128xf16>
// CHECK: call @to_bytes_f16(%[[B_f16]]) : (vector<128xf16>) -> vector<256xi8>
// CHECK: %[[A_bf16:[^ ]+]] = call @read_bf16({{.*}}) : (i64) -> vector<128xbf16>
// CHECK: %[[B_bf16:[^ ]+]] = call @step_bf16(%[[A_bf16]], {{.*}}) : (vector<128xbf16>, i64) -> vector<128xbf16>
// CHECK: call @to_bytes_bf16(%[[B_bf16]]) : (vector<128xbf16>) -> vector<256xi8>
// CHECK: %[[A_f32:[^ ]+]] = call @read_f32({{.*}}) : (i64) -> vector<64xf32>
// CHECK: %[[B_f32:[^ ]+]] = call @step_f32(%[[A_f32]], {{.*}}) : (vector<64xf32>, i64) -> vector<64xf32>
// CHECK: call @to_bytes_f32(%[[B_f32]]) : (vector<64xf32>) -> vector<256xi8>
// CHECK: %[[A_i8:[^ ]+]] = call @read_i8({{.*}}) : (i64) -> vector<256xi8>
// CHECK: %[[B_i8:[^ ]+]] = call @step_i8(%[[A_i8]], {{.*}}) : (vector<256xi8>, i64) -> vector<256xi8>
// CHECK: call @to_bytes_i8(%[[B_i8]]) : (vector<256xi8>) -> vector<256xi8>
// CHECK: %[[A_ui8:[^ ]+]] = call @read_ui8({{.*}}) : (i64) -> vector<256xi8>
// CHECK: %[[B_ui8:[^ ]+]] = call @step_ui8(%[[A_ui8]], {{.*}}) : (vector<256xi8>, i64) -> vector<256xi8>
// CHECK: call @to_bytes_ui8(%[[B_ui8]]) : (vector<256xi8>) -> vector<256xi8>
// CHECK: %[[A_i16:[^ ]+]] = call @read_i16({{.*}}) : (i64) -> vector<128xi16>
// CHECK: %[[B_i16:[^ ]+]] = call @step_i16(%[[A_i16]], {{.*}}) : (vector<128xi16>, i64) -> vector<128xi16>
// CHECK: call @to_bytes_i16(%[[B_i16]]) : (vector<128xi16>) -> vector<256xi8>
// CHECK: %[[A_ui16:[^ ]+]] = call @read_ui16({{.*}}) : (i64) -> vector<128xi16>
// CHECK: %[[B_ui16:[^ ]+]] = call @step_ui16(%[[A_ui16]], {{.*}}) : (vector<128xi16>, i64) -> vector<128xi16>
// CHECK: call @to_bytes_ui16(%[[B_ui16]]) : (vector<128xi16>) -> vector<256xi8>
// CHECK: %[[A_i32:[^ ]+]] = call @read_i32({{.*}}) : (i64) -> vector<64xi32>
// CHECK: %[[B_i32:[^ ]+]] = call @step_i32(%[[A_i32]], {{.*}}) : (vector<64xi32>, i64) -> vector<64xi32>
// CHECK: call @to_bytes_i32(%[[B_i32]]) : (vector<64xi32>) -> vector<256xi8>
// CHECK: %[[A_ui32:[^ ]+]] = call @read_ui32({{.*}}) : (i64) -> vector<64xi32>
// CHECK: %[[B_ui32:[^ ]+]] = call @step_ui32(%[[A_ui32]], {{.*}}) : (vector<64xi32>, i64) -> vector<64xi32>
// CHECK: call @to_bytes_ui32(%[[B_ui32]]) : (vector<64xi32>) -> vector<256xi8>
// CHECK: %[[A_i64:[^ ]+]] = call @read_i64({{.*}}) : (i64) -> vector<32xi64>
// CHECK: %[[B_i64:[^ ]+]] = call @step_i64(%[[A_i64]], {{.*}}) : (vector<32xi64>, i64) -> vector<32xi64>
// CHECK: call @to_bytes_i64(%[[B_i64]]) : (vector<32xi64>) -> vector<256xi8>
// CHECK: %[[A_ui64:[^ ]+]] = call @read_ui64({{.*}}) : (i64) -> vector<32xi64>
// CHECK: %[[B_ui64:[^ ]+]] = call @step_ui64(%[[A_ui64]], {{.*}}) : (vector<32xi64>, i64) -> vector<32xi64>
// CHECK: call @to_bytes_ui64(%[[B_ui64]]) : (vector<32xi64>) -> vector<256xi8>
