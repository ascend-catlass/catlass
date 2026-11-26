// RUN: %tla_compile %s --mlir-print-ir-after=tla-vector-region-simd -o %t 2>&1 | %filecheck %s
// RUN: %tla_compile %s --mlir-print-ir-after=convert-hivmave-to-ave-intrin -o %t 2>&1 | %filecheck %s --check-prefix=INTRIN

!short_f32 = !tla.tensor<!tla.layout<!tla.shape<64>, !tla.stride<1>, !tla.shape<3>, RowMajor>, !tla.coord<0>, !tla.ptr<f32, ub, 4>>
!f32_tile = !tla.tensor<!tla.layout<!tla.shape<64>, !tla.stride<1>, !tla.shape<64>, RowMajor>, !tla.coord<0>, !tla.ptr<f32, ub, 4>>
!f16_tile = !tla.tensor<!tla.layout<!tla.shape<128>, !tla.stride<1>, !tla.shape<128>, RowMajor>, !tla.coord<0>, !tla.ptr<f16, ub, 2>>
!i64_tile = !tla.tensor<!tla.layout<!tla.shape<32>, !tla.stride<1>, !tla.shape<32>, RowMajor>, !tla.coord<0>, !tla.ptr<i64, ub, 8>>
!u64_tile = !tla.tensor<!tla.layout<!tla.shape<32>, !tla.stride<1>, !tla.shape<32>, RowMajor>, !tla.coord<0>, !tla.ptr<ui64, ub, 8>>

// CHECK-DAG: func.func private @tla_axpy_f16(vector<128xf16>, vector<128xf16>, f16, vector<256xi1>) -> vector<128xf16>
// CHECK-DAG: func.func private @tla_axpy_f32(vector<64xf32>, vector<64xf32>, f32, vector<256xi1>) -> vector<64xf32>
// CHECK-DAG: func.func private @tla_axpy_i64(vector<32xi64>, vector<32xi64>, i64, vector<256xi1>) -> vector<32xi64>
// CHECK-DAG: func.func private @tla_axpy_u64(vector<32xi64>, vector<32xi64>, i64, vector<256xi1>) -> vector<32xi64>
// CHECK-DAG: func.func private @tla_sub_abs_i64(vector<32xi64>, vector<32xi64>, vector<256xi1>) -> vector<32xi64>
// CHECK-DAG: func.func private @tla_muls_cast_even(vector<64xf32>, f32, vector<256xi1>) -> vector<128xf16>
// CHECK-DAG: func.func private @tla_muls_cast_odd(vector<64xf32>, f32, vector<256xi1>) -> vector<128xf16>

module {
  func.func @compound_float(
      %short_mem: memref<64xf32, #hivm.address_space<ub>>,
      %full_mem: memref<64xf32, #hivm.address_space<ub>>,
      %half_mem: memref<128xf16, #hivm.address_space<ub>>,
      %scale: f32) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c3 = arith.constant 3 : index
    %c64 = arith.constant 64 : index
    %c128 = arith.constant 128 : index
    %short = tla.tensor_desc %short_mem shape [%c1, %c64, %c1, %c1] stride [%c64, %c1, %c1, %c1] origin_shape [%c1, %c3] coord [%c0, %c0] : memref<64xf32, #hivm.address_space<ub>> -> !short_f32
    %full = tla.tensor_desc %full_mem shape [%c1, %c64, %c1, %c1] stride [%c64, %c1, %c1, %c1] origin_shape [%c1, %c64] coord [%c0, %c0] : memref<64xf32, #hivm.address_space<ub>> -> !f32_tile
    %half = tla.tensor_desc %half_mem shape [%c1, %c128, %c1, %c1] stride [%c128, %c1, %c1, %c1] origin_shape [%c1, %c128] coord [%c0, %c0] : memref<128xf16, #hivm.address_space<ub>> -> !f16_tile
    "tla.vec.func"() ({
      %x = tla.load %short : !short_f32 -> !tla.vector<3xf32>
      %y = tla.load %full : !f32_tile -> !tla.vector<64xf32>
      %mask = "tla.create_mask"() {pattern = "VL16", dtype = f32} : () -> !tla.mask<64>
      %a = tla.axpy %x, %y, %scale mask %mask : !tla.vector<3xf32>, !tla.vector<64xf32>, f32 mask !tla.mask<64> -> !tla.vector<64xf32>
      %d = tla.sub_abs %a, %y mask %mask : !tla.vector<64xf32>, !tla.vector<64xf32> mask !tla.mask<64> -> !tla.vector<64xf32>
      %even = tla.muls_cast %x, %scale mask %mask part 0 : !tla.vector<3xf32> mask !tla.mask<64> -> !tla.vector<128xf16>
      %odd = tla.muls_cast %d, %scale mask %mask part 1 : !tla.vector<64xf32> mask !tla.mask<64> -> !tla.vector<128xf16>
      %half_scale = arith.constant 5.000000e-01 : f16
      %mask_half = "tla.create_mask"() {pattern = "ALL", dtype = f16} : () -> !tla.mask<128>
      %diff = tla.sub_abs %even, %odd mask %mask_half : !tla.vector<128xf16>, !tla.vector<128xf16> mask !tla.mask<128> -> !tla.vector<128xf16>
      %result = tla.axpy %diff, %diff, %half_scale mask %mask_half : !tla.vector<128xf16>, !tla.vector<128xf16>, f16 mask !tla.mask<128> -> !tla.vector<128xf16>
      tla.store %half, %result : !f16_tile, !tla.vector<128xf16>
    }) {mode = "simd"} : () -> ()
    return
  }

  func.func @compound_integer(
      %signed_mem: memref<32xi64, #hivm.address_space<ub>>,
      %unsigned_mem: memref<32xui64, #hivm.address_space<ub>>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c32 = arith.constant 32 : index
    %signed = tla.tensor_desc %signed_mem shape [%c1, %c32, %c1, %c1] stride [%c32, %c1, %c1, %c1] origin_shape [%c1, %c32] coord [%c0, %c0] : memref<32xi64, #hivm.address_space<ub>> -> !i64_tile
    %unsigned = tla.tensor_desc %unsigned_mem shape [%c1, %c32, %c1, %c1] stride [%c32, %c1, %c1, %c1] origin_shape [%c1, %c32] coord [%c0, %c0] : memref<32xui64, #hivm.address_space<ub>> -> !u64_tile
    "tla.vec.func"() ({
      %scale = arith.constant 5 : i64
      %s = tla.load %signed : !i64_tile -> !tla.vector<32xi64>
      %u = tla.load %unsigned : !u64_tile -> !tla.vector<32xui64>
      %mask = "tla.create_mask"() {pattern = "VL16", dtype = i64} : () -> !tla.mask<32>
      %diff = tla.sub_abs %s, %s mask %mask : !tla.vector<32xi64>, !tla.vector<32xi64> mask !tla.mask<32> -> !tla.vector<32xi64>
      %s_result = tla.axpy %diff, %s, %scale mask %mask : !tla.vector<32xi64>, !tla.vector<32xi64>, i64 mask !tla.mask<32> -> !tla.vector<32xi64>
      %u_result = tla.axpy %u, %u, %scale mask %mask : !tla.vector<32xui64>, !tla.vector<32xui64>, i64 mask !tla.mask<32> -> !tla.vector<32xui64>
      tla.store %signed, %s_result : !i64_tile, !tla.vector<32xi64>
      tla.store %unsigned, %u_result : !u64_tile, !tla.vector<32xui64>
    }) {mode = "simd"} : () -> ()
    return
  }
}

// CHECK-DAG: ave.hir.plt
// CHECK-DAG: ave.hir.preg.and
// CHECK-DAG: call @tla_axpy_f32
// CHECK-DAG: ave.hir.vabs_diff {{.*}} : vector<64xf32>, vector<256xi1>
// CHECK-DAG: call @tla_muls_cast_even
// CHECK-DAG: call @tla_muls_cast_odd
// CHECK-DAG: ave.hir.vabs_diff {{.*}} : vector<128xf16>, vector<256xi1>
// CHECK-DAG: call @tla_axpy_f16
// CHECK-DAG: call @tla_sub_abs_i64
// CHECK-DAG: call @tla_axpy_i64
// CHECK-DAG: call @tla_axpy_u64

// INTRIN-DAG: "hivm_regbaseintrins.intr.hivm.vabsdif.s.x"{{.*}} -> vector<64xf32>
// INTRIN-DAG: "hivm_regbaseintrins.intr.hivm.vabsdif.s.x"{{.*}} -> vector<128xf16>
