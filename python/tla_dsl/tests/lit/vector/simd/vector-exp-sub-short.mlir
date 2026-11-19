// RUN: %tla_compile %s --mlir-print-ir-after=tla-vector-region-simd -o %t 2>&1 | %filecheck %s --implicit-check-not=llvm.hivm.vexpdif

// Static valid prefixes bound the input-domain mask before native ZEROING.
// Results retain all 64 f32 lanes for both register slots.

!out = !tla.tensor<!tla.layout<!tla.shape<64>, !tla.stride<1>, !tla.shape<64>, RowMajor>, !tla.coord<0>, !tla.ptr<f32, ub, 4>>
!lhs0 = !tla.tensor<!tla.layout<!tla.shape<128>, !tla.stride<1>, !tla.shape<1>, RowMajor>, !tla.coord<0>, !tla.ptr<f16, ub, 2>>
!rhs0 = !tla.tensor<!tla.layout<!tla.shape<128>, !tla.stride<1>, !tla.shape<1>, RowMajor>, !tla.coord<0>, !tla.ptr<f16, ub, 2>>
!lhs1 = !tla.tensor<!tla.layout<!tla.shape<128>, !tla.stride<1>, !tla.shape<64>, RowMajor>, !tla.coord<0>, !tla.ptr<f16, ub, 2>>
!rhs1 = !tla.tensor<!tla.layout<!tla.shape<128>, !tla.stride<1>, !tla.shape<64>, RowMajor>, !tla.coord<0>, !tla.ptr<f16, ub, 2>>
!lhs2 = !tla.tensor<!tla.layout<!tla.shape<128>, !tla.stride<1>, !tla.shape<65>, RowMajor>, !tla.coord<0>, !tla.ptr<f16, ub, 2>>
!rhs2 = !tla.tensor<!tla.layout<!tla.shape<128>, !tla.stride<1>, !tla.shape<128>, RowMajor>, !tla.coord<0>, !tla.ptr<f16, ub, 2>>
!lhs3 = !tla.tensor<!tla.layout<!tla.shape<128>, !tla.stride<1>, !tla.shape<127>, RowMajor>, !tla.coord<0>, !tla.ptr<f16, ub, 2>>
!rhs3 = !tla.tensor<!tla.layout<!tla.shape<128>, !tla.stride<1>, !tla.shape<65>, RowMajor>, !tla.coord<0>, !tla.ptr<f16, ub, 2>>
!lhs4 = !tla.tensor<!tla.layout<!tla.shape<128>, !tla.stride<1>, !tla.shape<3>, RowMajor>, !tla.coord<0>, !tla.ptr<f16, ub, 2>>
!rhs4 = !tla.tensor<!tla.layout<!tla.shape<64>, !tla.stride<1>, !tla.shape<64>, RowMajor>, !tla.coord<0>, !tla.ptr<f32, ub, 4>>
!lhs5 = !tla.tensor<!tla.layout<!tla.shape<64>, !tla.stride<1>, !tla.shape<32>, RowMajor>, !tla.coord<0>, !tla.ptr<f32, ub, 4>>
!rhs5 = !tla.tensor<!tla.layout<!tla.shape<64>, !tla.stride<1>, !tla.shape<32>, RowMajor>, !tla.coord<0>, !tla.ptr<f32, ub, 4>>
!lhs6 = !tla.tensor<!tla.layout<!tla.shape<64>, !tla.stride<1>, !tla.shape<3>, RowMajor>, !tla.coord<0>, !tla.ptr<f32, ub, 4>>
!rhs6 = !tla.tensor<!tla.layout<!tla.shape<64>, !tla.stride<1>, !tla.shape<1>, RowMajor>, !tla.coord<0>, !tla.ptr<f32, ub, 4>>

!lhs7 = !tla.tensor<!tla.layout<!tla.shape<64>, !tla.stride<1>, !tla.shape<64>, RowMajor>, !tla.coord<0>, !tla.ptr<f32, ub, 4>>
!rhs7 = !tla.tensor<!tla.layout<!tla.shape<64>, !tla.stride<1>, !tla.shape<64>, RowMajor>, !tla.coord<0>, !tla.ptr<f32, ub, 4>>

module {
  func.func @half_one(
      %lhs_mem: memref<128xf16, #hivm.address_space<ub>>,
      %rhs_mem: memref<128xf16, #hivm.address_space<ub>>,
      %out0_mem: memref<64xf32, #hivm.address_space<ub>>,
      %out1_mem: memref<64xf32, #hivm.address_space<ub>>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %lhs_n = arith.constant 1 : index
    %rhs_n = arith.constant 1 : index
    %c64 = arith.constant 64 : index
    %c128 = arith.constant 128 : index
    %lhs = tla.tensor_desc %lhs_mem shape [%c1, %c128, %c1, %c1] stride [%lhs_n, %c1, %c1, %c1] origin_shape [%c1, %lhs_n] coord [%c0, %c0] : memref<128xf16, #hivm.address_space<ub>> -> !lhs0
    %rhs = tla.tensor_desc %rhs_mem shape [%c1, %c128, %c1, %c1] stride [%rhs_n, %c1, %c1, %c1] origin_shape [%c1, %rhs_n] coord [%c0, %c0] : memref<128xf16, #hivm.address_space<ub>> -> !rhs0
    %out0 = tla.tensor_desc %out0_mem shape [%c1, %c64, %c1, %c1] stride [%c64, %c1, %c1, %c1] origin_shape [%c1, %c64] coord [%c0, %c0] : memref<64xf32, #hivm.address_space<ub>> -> !out
    %out1 = tla.tensor_desc %out1_mem shape [%c1, %c64, %c1, %c1] stride [%c64, %c1, %c1, %c1] origin_shape [%c1, %c64] coord [%c0, %c0] : memref<64xf32, #hivm.address_space<ub>> -> !out
    "tla.vector"() ({
      "tla.vec.func"() ({
        %a = tla.load %lhs : !lhs0 -> !tla.vector<1xf16>
        %b_loaded = tla.load %rhs : !rhs0 -> !tla.vector<1xf16>
        %mask = "tla.create_mask"() {pattern = "ALL", dtype = f16} : () -> !tla.mask<128>
        %part0 = tla.exp_sub %a, %b_loaded mask %mask part 0 : !tla.vector<1xf16>, !tla.vector<1xf16> mask !tla.mask<128> -> !tla.vector<64xf32>
        %part1 = tla.exp_sub %a, %b_loaded mask %mask part 1 : !tla.vector<1xf16>, !tla.vector<1xf16> mask !tla.mask<128> -> !tla.vector<64xf32>
        tla.store %out0, %part0 : !out, !tla.vector<64xf32>
        tla.store %out1, %part1 : !out, !tla.vector<64xf32>
      }) {mode = "simd"} : () -> ()
    }) : () -> ()
    return
  }
  func.func @half_short(
      %lhs_mem: memref<128xf16, #hivm.address_space<ub>>,
      %rhs_mem: memref<128xf16, #hivm.address_space<ub>>,
      %out0_mem: memref<64xf32, #hivm.address_space<ub>>,
      %out1_mem: memref<64xf32, #hivm.address_space<ub>>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %lhs_n = arith.constant 64 : index
    %rhs_n = arith.constant 64 : index
    %c64 = arith.constant 64 : index
    %c128 = arith.constant 128 : index
    %lhs = tla.tensor_desc %lhs_mem shape [%c1, %c128, %c1, %c1] stride [%lhs_n, %c1, %c1, %c1] origin_shape [%c1, %lhs_n] coord [%c0, %c0] : memref<128xf16, #hivm.address_space<ub>> -> !lhs1
    %rhs = tla.tensor_desc %rhs_mem shape [%c1, %c128, %c1, %c1] stride [%rhs_n, %c1, %c1, %c1] origin_shape [%c1, %rhs_n] coord [%c0, %c0] : memref<128xf16, #hivm.address_space<ub>> -> !rhs1
    %out0 = tla.tensor_desc %out0_mem shape [%c1, %c64, %c1, %c1] stride [%c64, %c1, %c1, %c1] origin_shape [%c1, %c64] coord [%c0, %c0] : memref<64xf32, #hivm.address_space<ub>> -> !out
    %out1 = tla.tensor_desc %out1_mem shape [%c1, %c64, %c1, %c1] stride [%c64, %c1, %c1, %c1] origin_shape [%c1, %c64] coord [%c0, %c0] : memref<64xf32, #hivm.address_space<ub>> -> !out
    "tla.vector"() ({
      "tla.vec.func"() ({
        %a = tla.load %lhs : !lhs1 -> !tla.vector<64xf16>
        %b_loaded = tla.load %rhs : !rhs1 -> !tla.vector<64xf16>
        %mask = "tla.create_mask"() {pattern = "ALL", dtype = f16} : () -> !tla.mask<128>
        %part0 = tla.exp_sub %a, %b_loaded mask %mask part 0 : !tla.vector<64xf16>, !tla.vector<64xf16> mask !tla.mask<128> -> !tla.vector<64xf32>
        %part1 = tla.exp_sub %a, %b_loaded mask %mask part 1 : !tla.vector<64xf16>, !tla.vector<64xf16> mask !tla.mask<128> -> !tla.vector<64xf32>
        tla.store %out0, %part0 : !out, !tla.vector<64xf32>
        tla.store %out1, %part1 : !out, !tla.vector<64xf32>
      }) {mode = "simd"} : () -> ()
    }) : () -> ()
    return
  }
  func.func @half_short_full(
      %lhs_mem: memref<128xf16, #hivm.address_space<ub>>,
      %rhs_mem: memref<128xf16, #hivm.address_space<ub>>,
      %out0_mem: memref<64xf32, #hivm.address_space<ub>>,
      %out1_mem: memref<64xf32, #hivm.address_space<ub>>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %lhs_n = arith.constant 65 : index
    %rhs_n = arith.constant 128 : index
    %c64 = arith.constant 64 : index
    %c128 = arith.constant 128 : index
    %lhs = tla.tensor_desc %lhs_mem shape [%c1, %c128, %c1, %c1] stride [%lhs_n, %c1, %c1, %c1] origin_shape [%c1, %lhs_n] coord [%c0, %c0] : memref<128xf16, #hivm.address_space<ub>> -> !lhs2
    %rhs = tla.tensor_desc %rhs_mem shape [%c1, %c128, %c1, %c1] stride [%rhs_n, %c1, %c1, %c1] origin_shape [%c1, %rhs_n] coord [%c0, %c0] : memref<128xf16, #hivm.address_space<ub>> -> !rhs2
    %out0 = tla.tensor_desc %out0_mem shape [%c1, %c64, %c1, %c1] stride [%c64, %c1, %c1, %c1] origin_shape [%c1, %c64] coord [%c0, %c0] : memref<64xf32, #hivm.address_space<ub>> -> !out
    %out1 = tla.tensor_desc %out1_mem shape [%c1, %c64, %c1, %c1] stride [%c64, %c1, %c1, %c1] origin_shape [%c1, %c64] coord [%c0, %c0] : memref<64xf32, #hivm.address_space<ub>> -> !out
    "tla.vector"() ({
      "tla.vec.func"() ({
        %a = tla.load %lhs : !lhs2 -> !tla.vector<65xf16>
        %b_loaded = tla.load %rhs : !rhs2 -> !tla.vector<128xf16>
        %mask = "tla.create_mask"() {pattern = "ALL", dtype = f16} : () -> !tla.mask<128>
        %part0 = tla.exp_sub %a, %b_loaded mask %mask part 0 : !tla.vector<65xf16>, !tla.vector<128xf16> mask !tla.mask<128> -> !tla.vector<64xf32>
        %part1 = tla.exp_sub %a, %b_loaded mask %mask part 1 : !tla.vector<65xf16>, !tla.vector<128xf16> mask !tla.mask<128> -> !tla.vector<64xf32>
        tla.store %out0, %part0 : !out, !tla.vector<64xf32>
        tla.store %out1, %part1 : !out, !tla.vector<64xf32>
      }) {mode = "simd"} : () -> ()
    }) : () -> ()
    return
  }
  func.func @half_different(
      %lhs_mem: memref<128xf16, #hivm.address_space<ub>>,
      %rhs_mem: memref<128xf16, #hivm.address_space<ub>>,
      %out0_mem: memref<64xf32, #hivm.address_space<ub>>,
      %out1_mem: memref<64xf32, #hivm.address_space<ub>>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %lhs_n = arith.constant 127 : index
    %rhs_n = arith.constant 65 : index
    %c64 = arith.constant 64 : index
    %c128 = arith.constant 128 : index
    %lhs = tla.tensor_desc %lhs_mem shape [%c1, %c128, %c1, %c1] stride [%lhs_n, %c1, %c1, %c1] origin_shape [%c1, %lhs_n] coord [%c0, %c0] : memref<128xf16, #hivm.address_space<ub>> -> !lhs3
    %rhs = tla.tensor_desc %rhs_mem shape [%c1, %c128, %c1, %c1] stride [%rhs_n, %c1, %c1, %c1] origin_shape [%c1, %rhs_n] coord [%c0, %c0] : memref<128xf16, #hivm.address_space<ub>> -> !rhs3
    %out0 = tla.tensor_desc %out0_mem shape [%c1, %c64, %c1, %c1] stride [%c64, %c1, %c1, %c1] origin_shape [%c1, %c64] coord [%c0, %c0] : memref<64xf32, #hivm.address_space<ub>> -> !out
    %out1 = tla.tensor_desc %out1_mem shape [%c1, %c64, %c1, %c1] stride [%c64, %c1, %c1, %c1] origin_shape [%c1, %c64] coord [%c0, %c0] : memref<64xf32, #hivm.address_space<ub>> -> !out
    "tla.vector"() ({
      "tla.vec.func"() ({
        %a = tla.load %lhs : !lhs3 -> !tla.vector<127xf16>
        %b_loaded = tla.load %rhs : !rhs3 -> !tla.vector<65xf16>
        %mask = "tla.create_mask"() {pattern = "ALL", dtype = f16} : () -> !tla.mask<128>
        %part0 = tla.exp_sub %a, %b_loaded mask %mask part 0 : !tla.vector<127xf16>, !tla.vector<65xf16> mask !tla.mask<128> -> !tla.vector<64xf32>
        %part1 = tla.exp_sub %a, %b_loaded mask %mask part 1 : !tla.vector<127xf16>, !tla.vector<65xf16> mask !tla.mask<128> -> !tla.vector<64xf32>
        tla.store %out0, %part0 : !out, !tla.vector<64xf32>
        tla.store %out1, %part1 : !out, !tla.vector<64xf32>
      }) {mode = "simd"} : () -> ()
    }) : () -> ()
    return
  }
  func.func @half_unknown(
      %lhs_mem: memref<128xf16, #hivm.address_space<ub>>,
      %rhs_mem: memref<64xf32, #hivm.address_space<ub>>,
      %out0_mem: memref<64xf32, #hivm.address_space<ub>>,
      %out1_mem: memref<64xf32, #hivm.address_space<ub>>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %lhs_n = arith.constant 3 : index
    %rhs_n = arith.constant 64 : index
    %c64 = arith.constant 64 : index
    %c128 = arith.constant 128 : index
    %lhs = tla.tensor_desc %lhs_mem shape [%c1, %c128, %c1, %c1] stride [%lhs_n, %c1, %c1, %c1] origin_shape [%c1, %lhs_n] coord [%c0, %c0] : memref<128xf16, #hivm.address_space<ub>> -> !lhs4
    %rhs = tla.tensor_desc %rhs_mem shape [%c1, %c64, %c1, %c1] stride [%rhs_n, %c1, %c1, %c1] origin_shape [%c1, %rhs_n] coord [%c0, %c0] : memref<64xf32, #hivm.address_space<ub>> -> !rhs4
    %out0 = tla.tensor_desc %out0_mem shape [%c1, %c64, %c1, %c1] stride [%c64, %c1, %c1, %c1] origin_shape [%c1, %c64] coord [%c0, %c0] : memref<64xf32, #hivm.address_space<ub>> -> !out
    %out1 = tla.tensor_desc %out1_mem shape [%c1, %c64, %c1, %c1] stride [%c64, %c1, %c1, %c1] origin_shape [%c1, %c64] coord [%c0, %c0] : memref<64xf32, #hivm.address_space<ub>> -> !out
    "tla.vector"() ({
      "tla.vec.func"() ({
        %a = tla.load %lhs : !lhs4 -> !tla.vector<3xf16>
        %b_loaded = tla.load %rhs : !rhs4 -> !tla.vector<64xf32>
        %b = tla.cast %b_loaded, [0, 0, 0] : !tla.vector<64xf32> -> !tla.vector<?xf16>
        %mask = "tla.create_mask"() {pattern = "ALL", dtype = f16} : () -> !tla.mask<128>
        %part0 = tla.exp_sub %a, %b mask %mask part 0 : !tla.vector<3xf16>, !tla.vector<?xf16> mask !tla.mask<128> -> !tla.vector<64xf32>
        %part1 = tla.exp_sub %a, %b mask %mask part 1 : !tla.vector<3xf16>, !tla.vector<?xf16> mask !tla.mask<128> -> !tla.vector<64xf32>
        tla.store %out0, %part0 : !out, !tla.vector<64xf32>
        tla.store %out1, %part1 : !out, !tla.vector<64xf32>
      }) {mode = "simd"} : () -> ()
    }) : () -> ()
    return
  }
  func.func @float_short(
      %lhs_mem: memref<64xf32, #hivm.address_space<ub>>,
      %rhs_mem: memref<64xf32, #hivm.address_space<ub>>,
      %out0_mem: memref<64xf32, #hivm.address_space<ub>>,
      %out1_mem: memref<64xf32, #hivm.address_space<ub>>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %lhs_n = arith.constant 32 : index
    %rhs_n = arith.constant 32 : index
    %c64 = arith.constant 64 : index
    %c128 = arith.constant 128 : index
    %lhs = tla.tensor_desc %lhs_mem shape [%c1, %c64, %c1, %c1] stride [%lhs_n, %c1, %c1, %c1] origin_shape [%c1, %lhs_n] coord [%c0, %c0] : memref<64xf32, #hivm.address_space<ub>> -> !lhs5
    %rhs = tla.tensor_desc %rhs_mem shape [%c1, %c64, %c1, %c1] stride [%rhs_n, %c1, %c1, %c1] origin_shape [%c1, %rhs_n] coord [%c0, %c0] : memref<64xf32, #hivm.address_space<ub>> -> !rhs5
    %out0 = tla.tensor_desc %out0_mem shape [%c1, %c64, %c1, %c1] stride [%c64, %c1, %c1, %c1] origin_shape [%c1, %c64] coord [%c0, %c0] : memref<64xf32, #hivm.address_space<ub>> -> !out
    %out1 = tla.tensor_desc %out1_mem shape [%c1, %c64, %c1, %c1] stride [%c64, %c1, %c1, %c1] origin_shape [%c1, %c64] coord [%c0, %c0] : memref<64xf32, #hivm.address_space<ub>> -> !out
    "tla.vector"() ({
      "tla.vec.func"() ({
        %a = tla.load %lhs : !lhs5 -> !tla.vector<32xf32>
        %b_loaded = tla.load %rhs : !rhs5 -> !tla.vector<32xf32>
        %mask = "tla.create_mask"() {pattern = "ALL", dtype = f32} : () -> !tla.mask<64>
        %part0 = tla.exp_sub %a, %b_loaded mask %mask part 0 : !tla.vector<32xf32>, !tla.vector<32xf32> mask !tla.mask<64> -> !tla.vector<64xf32>
        %part1 = tla.exp_sub %a, %b_loaded mask %mask part 1 : !tla.vector<32xf32>, !tla.vector<32xf32> mask !tla.mask<64> -> !tla.vector<64xf32>
        tla.store %out0, %part0 : !out, !tla.vector<64xf32>
        tla.store %out1, %part1 : !out, !tla.vector<64xf32>
      }) {mode = "simd"} : () -> ()
    }) : () -> ()
    return
  }
  func.func @float_different(
      %lhs_mem: memref<64xf32, #hivm.address_space<ub>>,
      %rhs_mem: memref<64xf32, #hivm.address_space<ub>>,
      %out0_mem: memref<64xf32, #hivm.address_space<ub>>,
      %out1_mem: memref<64xf32, #hivm.address_space<ub>>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %lhs_n = arith.constant 3 : index
    %rhs_n = arith.constant 1 : index
    %c64 = arith.constant 64 : index
    %c128 = arith.constant 128 : index
    %lhs = tla.tensor_desc %lhs_mem shape [%c1, %c64, %c1, %c1] stride [%lhs_n, %c1, %c1, %c1] origin_shape [%c1, %lhs_n] coord [%c0, %c0] : memref<64xf32, #hivm.address_space<ub>> -> !lhs6
    %rhs = tla.tensor_desc %rhs_mem shape [%c1, %c64, %c1, %c1] stride [%rhs_n, %c1, %c1, %c1] origin_shape [%c1, %rhs_n] coord [%c0, %c0] : memref<64xf32, #hivm.address_space<ub>> -> !rhs6
    %out0 = tla.tensor_desc %out0_mem shape [%c1, %c64, %c1, %c1] stride [%c64, %c1, %c1, %c1] origin_shape [%c1, %c64] coord [%c0, %c0] : memref<64xf32, #hivm.address_space<ub>> -> !out
    %out1 = tla.tensor_desc %out1_mem shape [%c1, %c64, %c1, %c1] stride [%c64, %c1, %c1, %c1] origin_shape [%c1, %c64] coord [%c0, %c0] : memref<64xf32, #hivm.address_space<ub>> -> !out
    "tla.vector"() ({
      "tla.vec.func"() ({
        %a = tla.load %lhs : !lhs6 -> !tla.vector<3xf32>
        %b_loaded = tla.load %rhs : !rhs6 -> !tla.vector<1xf32>
        %mask = "tla.create_mask"() {pattern = "ALL", dtype = f32} : () -> !tla.mask<64>
        %part0 = tla.exp_sub %a, %b_loaded mask %mask part 0 : !tla.vector<3xf32>, !tla.vector<1xf32> mask !tla.mask<64> -> !tla.vector<64xf32>
        %part1 = tla.exp_sub %a, %b_loaded mask %mask part 1 : !tla.vector<3xf32>, !tla.vector<1xf32> mask !tla.mask<64> -> !tla.vector<64xf32>
        tla.store %out0, %part0 : !out, !tla.vector<64xf32>
        tla.store %out1, %part1 : !out, !tla.vector<64xf32>
      }) {mode = "simd"} : () -> ()
    }) : () -> ()
    return
  }
  func.func @half_both_unknown(
      %lhs_mem: memref<64xf32, #hivm.address_space<ub>>,
      %rhs_mem: memref<64xf32, #hivm.address_space<ub>>,
      %out0_mem: memref<64xf32, #hivm.address_space<ub>>,
      %out1_mem: memref<64xf32, #hivm.address_space<ub>>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %lhs_n = arith.constant 64 : index
    %rhs_n = arith.constant 64 : index
    %c64 = arith.constant 64 : index
    %c128 = arith.constant 128 : index
    %lhs = tla.tensor_desc %lhs_mem shape [%c1, %c64, %c1, %c1] stride [%lhs_n, %c1, %c1, %c1] origin_shape [%c1, %lhs_n] coord [%c0, %c0] : memref<64xf32, #hivm.address_space<ub>> -> !lhs7
    %rhs = tla.tensor_desc %rhs_mem shape [%c1, %c64, %c1, %c1] stride [%rhs_n, %c1, %c1, %c1] origin_shape [%c1, %rhs_n] coord [%c0, %c0] : memref<64xf32, #hivm.address_space<ub>> -> !rhs7
    %out0 = tla.tensor_desc %out0_mem shape [%c1, %c64, %c1, %c1] stride [%c64, %c1, %c1, %c1] origin_shape [%c1, %c64] coord [%c0, %c0] : memref<64xf32, #hivm.address_space<ub>> -> !out
    %out1 = tla.tensor_desc %out1_mem shape [%c1, %c64, %c1, %c1] stride [%c64, %c1, %c1, %c1] origin_shape [%c1, %c64] coord [%c0, %c0] : memref<64xf32, #hivm.address_space<ub>> -> !out
    "tla.vector"() ({
      "tla.vec.func"() ({
        %a_loaded = tla.load %lhs : !lhs7 -> !tla.vector<64xf32>
        %a = tla.cast %a_loaded, [0, 0, 0] : !tla.vector<64xf32> -> !tla.vector<?xf16>
        %b_loaded = tla.load %rhs : !rhs7 -> !tla.vector<64xf32>
        %b = tla.cast %b_loaded, [0, 0, 0] : !tla.vector<64xf32> -> !tla.vector<?xf16>
        %mask = "tla.create_mask"() {pattern = "ALL", dtype = f16} : () -> !tla.mask<128>
        %part0 = tla.exp_sub %a, %b mask %mask part 0 : !tla.vector<?xf16>, !tla.vector<?xf16> mask !tla.mask<128> -> !tla.vector<64xf32>
        %part1 = tla.exp_sub %a, %b mask %mask part 1 : !tla.vector<?xf16>, !tla.vector<?xf16> mask !tla.mask<128> -> !tla.vector<64xf32>
        tla.store %out0, %part0 : !out, !tla.vector<64xf32>
        tla.store %out1, %part1 : !out, !tla.vector<64xf32>
      }) {mode = "simd"} : () -> ()
    }) : () -> ()
    return
  }
}

// CHECK-LABEL: func.func private @vector_region_0(
// CHECK: ave.hir.vload
// CHECK: ave.hir.vload
// CHECK: %[[COUNT0:[a-zA-Z0-9_]+]] = arith.constant 1 : index
// CHECK-NEXT: %[[PREFIX0:[a-zA-Z0-9_]+]], %{{[a-zA-Z0-9_]+}} = ave.hir.plt %[[COUNT0]] {functionType = #ave.func_dist_type<pb16>}
// CHECK-NEXT: %[[ACTIVE0:[a-zA-Z0-9_]+]] = ave.hir.preg.and <b16> %{{[a-zA-Z0-9_]+}}, %[[PREFIX0]], %{{[a-zA-Z0-9_]+}}
// CHECK: call @exp_sub_f16_even(%{{[a-zA-Z0-9_]+}}, %{{[a-zA-Z0-9_]+}}, %[[ACTIVE0]])
// CHECK: ave.hir.preg.and <b16>
// CHECK: call @exp_sub_f16_odd
// CHECK-LABEL: func.func private @vector_region_1(
// CHECK: ave.hir.vload
// CHECK: ave.hir.vload
// CHECK: %[[COUNT1:[a-zA-Z0-9_]+]] = arith.constant 64 : index
// CHECK-NEXT: %[[PREFIX1:[a-zA-Z0-9_]+]], %{{[a-zA-Z0-9_]+}} = ave.hir.plt %[[COUNT1]] {functionType = #ave.func_dist_type<pb16>}
// CHECK-NEXT: %[[ACTIVE1:[a-zA-Z0-9_]+]] = ave.hir.preg.and <b16> %{{[a-zA-Z0-9_]+}}, %[[PREFIX1]], %{{[a-zA-Z0-9_]+}}
// CHECK: call @exp_sub_f16_even(%{{[a-zA-Z0-9_]+}}, %{{[a-zA-Z0-9_]+}}, %[[ACTIVE1]])
// CHECK: ave.hir.preg.and <b16>
// CHECK: call @exp_sub_f16_odd
// CHECK-LABEL: func.func private @vector_region_2(
// CHECK: ave.hir.vload
// CHECK: ave.hir.vload
// CHECK: %[[COUNT2:[a-zA-Z0-9_]+]] = arith.constant 65 : index
// CHECK-NEXT: %[[PREFIX2:[a-zA-Z0-9_]+]], %{{[a-zA-Z0-9_]+}} = ave.hir.plt %[[COUNT2]] {functionType = #ave.func_dist_type<pb16>}
// CHECK-NEXT: %[[ACTIVE2:[a-zA-Z0-9_]+]] = ave.hir.preg.and <b16> %{{[a-zA-Z0-9_]+}}, %[[PREFIX2]], %{{[a-zA-Z0-9_]+}}
// CHECK: call @exp_sub_f16_even(%{{[a-zA-Z0-9_]+}}, %{{[a-zA-Z0-9_]+}}, %[[ACTIVE2]])
// CHECK: ave.hir.preg.and <b16>
// CHECK: call @exp_sub_f16_odd
// CHECK-LABEL: func.func private @vector_region_3(
// CHECK: ave.hir.vload
// CHECK: ave.hir.vload
// CHECK: %[[COUNT3:[a-zA-Z0-9_]+]] = arith.constant 65 : index
// CHECK-NEXT: %[[PREFIX3:[a-zA-Z0-9_]+]], %{{[a-zA-Z0-9_]+}} = ave.hir.plt %[[COUNT3]] {functionType = #ave.func_dist_type<pb16>}
// CHECK-NEXT: %[[ACTIVE3:[a-zA-Z0-9_]+]] = ave.hir.preg.and <b16> %{{[a-zA-Z0-9_]+}}, %[[PREFIX3]], %{{[a-zA-Z0-9_]+}}
// CHECK: call @exp_sub_f16_even(%{{[a-zA-Z0-9_]+}}, %{{[a-zA-Z0-9_]+}}, %[[ACTIVE3]])
// CHECK: ave.hir.preg.and <b16>
// CHECK: call @exp_sub_f16_odd
// CHECK-LABEL: func.func private @vector_region_4(
// CHECK: ave.hir.vload
// CHECK: ave.hir.vload
// CHECK: %[[COUNT4:[a-zA-Z0-9_]+]] = arith.constant 3 : index
// CHECK-NEXT: %[[PREFIX4:[a-zA-Z0-9_]+]], %{{[a-zA-Z0-9_]+}} = ave.hir.plt %[[COUNT4]] {functionType = #ave.func_dist_type<pb16>}
// CHECK-NEXT: %[[ACTIVE4:[a-zA-Z0-9_]+]] = ave.hir.preg.and <b16> %{{[a-zA-Z0-9_]+}}, %[[PREFIX4]], %{{[a-zA-Z0-9_]+}}
// CHECK: call @exp_sub_f16_even(%{{[a-zA-Z0-9_]+}}, %{{[a-zA-Z0-9_]+}}, %[[ACTIVE4]])
// CHECK: ave.hir.preg.and <b16>
// CHECK: call @exp_sub_f16_odd
// CHECK-LABEL: func.func private @vector_region_5(
// CHECK: ave.hir.vload
// CHECK: ave.hir.vload
// CHECK: %[[COUNT5:[a-zA-Z0-9_]+]] = arith.constant 32 : index
// CHECK-NEXT: %[[PREFIX5:[a-zA-Z0-9_]+]], %{{[a-zA-Z0-9_]+}} = ave.hir.plt %[[COUNT5]] {functionType = #ave.func_dist_type<pb32>}
// CHECK-NEXT: %[[ACTIVE5:[a-zA-Z0-9_]+]] = ave.hir.preg.and <b32> %{{[a-zA-Z0-9_]+}}, %[[PREFIX5]], %{{[a-zA-Z0-9_]+}}
// CHECK: call @exp_sub_f32(%{{[a-zA-Z0-9_]+}}, %{{[a-zA-Z0-9_]+}}, %[[ACTIVE5]])
// CHECK: ave.hir.preg.and <b32>
// CHECK: call @exp_sub_f32
// CHECK-LABEL: func.func private @vector_region_6(
// CHECK: ave.hir.vload
// CHECK: ave.hir.vload
// CHECK: %[[COUNT6:[a-zA-Z0-9_]+]] = arith.constant 1 : index
// CHECK-NEXT: %[[PREFIX6:[a-zA-Z0-9_]+]], %{{[a-zA-Z0-9_]+}} = ave.hir.plt %[[COUNT6]] {functionType = #ave.func_dist_type<pb32>}
// CHECK-NEXT: %[[ACTIVE6:[a-zA-Z0-9_]+]] = ave.hir.preg.and <b32> %{{[a-zA-Z0-9_]+}}, %[[PREFIX6]], %{{[a-zA-Z0-9_]+}}
// CHECK: call @exp_sub_f32(%{{[a-zA-Z0-9_]+}}, %{{[a-zA-Z0-9_]+}}, %[[ACTIVE6]])
// CHECK: ave.hir.preg.and <b32>
// CHECK: call @exp_sub_f32
// CHECK-LABEL: func.func private @vector_region_7(
// CHECK-NOT: ave.hir.plt
// CHECK-NOT: ave.hir.preg.and
// CHECK: call @exp_sub_f16_even
// CHECK-NOT: ave.hir.plt
// CHECK-NOT: ave.hir.preg.and
// CHECK: call @exp_sub_f16_odd
// CHECK-NOT: ave.hir.plt
// CHECK-NOT: ave.hir.preg.and
