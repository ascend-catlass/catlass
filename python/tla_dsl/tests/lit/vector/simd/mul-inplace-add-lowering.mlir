// RUN: %python %S/Inputs/generate_mul_inplace_add_ir.py f16 | %tla_compile - -o - | %filecheck %s
// RUN: %python %S/Inputs/generate_mul_inplace_add_ir.py bf16 | %tla_compile - -o - | %filecheck %s
// RUN: %python %S/Inputs/generate_mul_inplace_add_ir.py f32 | %tla_compile - -o - | %filecheck %s
// RUN: %python %S/Inputs/generate_mul_inplace_add_ir.py i16 | %tla_compile - -o - | %filecheck %s
// RUN: %python %S/Inputs/generate_mul_inplace_add_ir.py ui16 | %tla_compile - -o - | %filecheck %s
// RUN: %python %S/Inputs/generate_mul_inplace_add_ir.py i32 | %tla_compile - -o - | %filecheck %s
// RUN: %python %S/Inputs/generate_mul_inplace_add_ir.py ui32 | %tla_compile - -o - | %filecheck %s
// RUN: %python %S/Inputs/generate_mul_inplace_add_ir.py i64 | %tla_compile - -o - | %filecheck %s --check-prefix=B64 -D'$SYMBOL'=tla_mul_inplace_add_i64 --implicit-check-not=hivm.vmula
// RUN: %python %S/Inputs/generate_mul_inplace_add_ir.py ui64 | %tla_compile - -o - | %filecheck %s --check-prefix=B64 -D'$SYMBOL'=tla_mul_inplace_add_u64 --implicit-check-not=hivm.vmula

// CHECK-LABEL: func.func private @vector_region_
// CHECK: %[[ACC:[a-zA-Z0-9_]+]] = {{.*}}hivm.vld
// CHECK: %[[LHS:[a-zA-Z0-9_]+]] = {{.*}}hivm.vld
// CHECK: %[[RHS:[a-zA-Z0-9_]+]] = {{.*}}hivm.vld
// CHECK: hivm.vmula.s.m{{.*}}(%[[ACC]], %[[LHS]], %[[RHS]],
// CHECK-NOT: tla.mul_inplace_add

// B64-LABEL: func.func private @vector_region_
// B64: call @[[$SYMBOL]]({{.*}}) : (vector<32xi64>, vector<32xi64>, vector<32xi64>, vector<256xi1>) -> vector<32xi64>
