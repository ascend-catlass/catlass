// RUN: %tla_compile %s -o %t --mlir-print-ir-after=tla-lower-ptr 2>&1 | %filecheck %s

module {
  func.func @typed_alloc_ptr() {
    %0 = "tla.alloc_ptr"() {size_bytes = 512 : i64} : () -> !tla.ptr<f16, l1, 512>
    %1 = "tla.alloc_ptr"() {size_bytes = 512 : i64} : () -> !tla.ptr<f32, ub, 256>
    %2 = "tla.alloc_ptr"() {size_bytes = 128 : i64} : () -> !tla.ptr<i16, l0c, 1024>
    func.call @sink(%0, %1, %2) : (!tla.ptr<f16, l1, 512>, !tla.ptr<f32, ub, 256>, !tla.ptr<i16, l0c, 1024>) -> ()
    func.return
  }
  func.func private @sink(!tla.ptr<f16, l1, 512>, !tla.ptr<f32, ub, 256>, !tla.ptr<i16, l0c, 1024>)
}

// UB is reserved through a symbol so that the object reports the bytes as
// statically allocated; the runtime reads that when deciding where the SIMT
// Data Cache starts, and a bare address would leave it reporting nothing. The
// global is private because an externally visible one without an initializer is
// only a declaration, which reserves no space.
// CHECK: llvm.mlir.global private @tla_ub_scratch() {addr_space = 6 : i32} : !llvm.array<512 x i8>

// CHECK-LABEL: func.func @typed_alloc_ptr
// The UB pointer is derived from that symbol rather than from a literal, which
// is what keeps the reservation alive through to the linker.
// CHECK-DAG: llvm.mlir.addressof @tla_ub_scratch
// CHECK-DAG: llvm.ptrtoint

// L1 and L0C keep bare addresses: UB is the only space the runtime partitions
// against the Data Cache, so it is the only one the object must account for.
// Each address space owns an independent offset range, so both start at zero.
// CHECK-DAG: arith.constant {{.*}}0 : i64
// CHECK-DAG: arith.constant {{.*}}0 : i64

// Function ABI conversion happens atomically with the pointer producers.
// CHECK: call @sink({{.*}}) : (i64, i64, i64) -> ()
// CHECK-NOT: !tla.ptr
// CHECK-NOT: tla.alloc_ptr
// CHECK: func.func private @sink(i64, i64, i64)
