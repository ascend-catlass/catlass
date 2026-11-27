// RUN: %tla_compile %s --mlir-print-ir-after=tla-vector-region-simd -o %t 2>&1 | %filecheck %s --implicit-check-not=tla.unpack

!t32 = !tla.tensor<!tla.layout<!tla.shape<64>, !tla.stride<1>, !tla.shape<64>, RowMajor>, !tla.coord<0>, !tla.ptr<i32, ub, 4>>
!t16 = !tla.tensor<!tla.layout<!tla.shape<128>, !tla.stride<1>, !tla.shape<128>, RowMajor>, !tla.coord<0>, !tla.ptr<i16, ub, 2>>

module {
  func.func @unpack_mask(%out32: memref<64xi32, #hivm.address_space<ub>>, %out16: memref<128xi16, #hivm.address_space<ub>>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c64 = arith.constant 64 : index
    %c128 = arith.constant 128 : index
    %t32 = tla.tensor_desc %out32 shape [%c1, %c64, %c1, %c1] stride [%c64, %c1, %c1, %c1] origin_shape [%c1, %c64] coord [%c0, %c0] : memref<64xi32, #hivm.address_space<ub>> -> !t32
    %t16 = tla.tensor_desc %out16 shape [%c1, %c128, %c1, %c1] stride [%c128, %c1, %c1, %c1] origin_shape [%c1, %c128] coord [%c0, %c0] : memref<128xi16, #hivm.address_space<ub>> -> !t16
    "tla.vec.func"() ({
      %v32 = tla.load %t32 : !t32 -> !tla.vector<64xi32>
      %v16 = tla.load %t16 : !t16 -> !tla.vector<128xi16>
      %m128 = "tla.create_mask"() {pattern = "H", dtype = i16} : () -> !tla.mask<128>
      %m256 = "tla.create_mask"() {pattern = "Q", dtype = i8} : () -> !tla.mask<256>
      %m64 = tla.unpack %m128, <low> : !tla.mask<128> -> !tla.mask<64>
      %res128 = tla.unpack %m256, <high> : !tla.mask<256> -> !tla.mask<128>
      %out_i32 = "tla.where"(%m64, %v32, %v32) : (!tla.mask<64>, !tla.vector<64xi32>, !tla.vector<64xi32>) -> !tla.vector<64xi32>
      %out_i16 = "tla.where"(%res128, %v16, %v16) : (!tla.mask<128>, !tla.vector<128xi16>, !tla.vector<128xi16>) -> !tla.vector<128xi16>
      tla.store %t32, %out_i32 : !t32, !tla.vector<64xi32>
      tla.store %t16, %out_i16 : !t16, !tla.vector<128xi16>
    }) {mode = "simd"} : () -> ()
    return
  }
}

// CHECK-LABEL: func.func private @vector_region_
// CHECK: %[[LOW:.*]] = arith.constant 0 : i32
// CHECK: %[[MASK64:.*]] = "hivm_regbaseintrins.intr.hivm.punpack"({{.*}}, %[[LOW]]) : (vector<256xi1>, i32) -> vector<256xi1>
// CHECK: %[[HIGH:.*]] = arith.constant 1 : i32
// CHECK: %[[MASK128:.*]] = "hivm_regbaseintrins.intr.hivm.punpack"({{.*}}, %[[HIGH]]) : (vector<256xi1>, i32) -> vector<256xi1>
// CHECK: ave.hir.vsel %[[MASK64]]
// CHECK: ave.hir.vsel %[[MASK128]]
