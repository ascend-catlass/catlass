// RUN: %tla_compile %s --mlir-print-ir-after=tla-vector-region-simd -o %t 2>&1 | %filecheck %s
// PACK4 predicates are expressed in source lanes, including short tails.
!s_byte = !tla.tensor<!tla.layout<!tla.shape<256>, !tla.stride<1>, !tla.shape<256>, RowMajor>, !tla.coord<0>, !tla.ptr<i8, ub, 32>>
!d_byte = !tla.tensor<!tla.layout<!tla.shape<13>, !tla.stride<1>, !tla.shape<13>, RowMajor>, !tla.coord<0>, !tla.ptr<i8, ub, 32>>
!s_half = !tla.tensor<!tla.layout<!tla.shape<128>, !tla.stride<1>, !tla.shape<128>, RowMajor>, !tla.coord<0>, !tla.ptr<i16, ub, 32>>
!d_half = !tla.tensor<!tla.layout<!tla.shape<13>, !tla.stride<1>, !tla.shape<13>, RowMajor>, !tla.coord<0>, !tla.ptr<i8, ub, 32>>
!s_word = !tla.tensor<!tla.layout<!tla.shape<64>, !tla.stride<1>, !tla.shape<64>, RowMajor>, !tla.coord<0>, !tla.ptr<i32, ub, 32>>
!d_word = !tla.tensor<!tla.layout<!tla.shape<13>, !tla.stride<1>, !tla.shape<13>, RowMajor>, !tla.coord<0>, !tla.ptr<i8, ub, 32>>
!s_full = !tla.tensor<!tla.layout<!tla.shape<256>, !tla.stride<1>, !tla.shape<256>, RowMajor>, !tla.coord<0>, !tla.ptr<f8E5M2, ub, 32>>
!d_full = !tla.tensor<!tla.layout<!tla.shape<64>, !tla.stride<1>, !tla.shape<64>, RowMajor>, !tla.coord<0>, !tla.ptr<i8, ub, 32>>
!us_e4m3 = !tla.tensor<!tla.layout<!tla.shape<64>, !tla.stride<1>, !tla.shape<64>, RowMajor>, !tla.coord<0>, !tla.ptr<f8E4M3FN, ub, 32>>
!ud_e4m3 = !tla.tensor<!tla.layout<!tla.shape<256>, !tla.stride<1>, !tla.shape<256>, RowMajor>, !tla.coord<0>, !tla.ptr<f8E4M3FN, ub, 32>>

module {
// CHECK-LABEL: func.func @pack4_byte
// CHECK-LABEL: func.func private @vector_region_
// CHECK: %[[N_BYTE:.*]] = arith.constant 52 : index
// CHECK: ave.hir.plt %[[N_BYTE]]
// CHECK: ave.hir.masked_store <PK4_B32>
// CHECK-SAME: vector<256xi8>
  func.func @pack4_byte(%sm: memref<256xi8, #hivm.address_space<ub>>, %dm: memref<13xi8, #hivm.address_space<ub>>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %cs = arith.constant 256 : index
    %cd = arith.constant 13 : index
    %s = tla.tensor_desc %sm shape [%c1, %cs, %c1, %c1] stride [%cs, %c1, %c1, %c1] origin_shape [%c1, %cs] coord [%c0, %c0] : memref<256xi8, #hivm.address_space<ub>> -> !s_byte
    %d = tla.tensor_desc %dm shape [%c1, %cd, %c1, %c1] stride [%cd, %c1, %c1, %c1] origin_shape [%c1, %cd] coord [%c0, %c0] : memref<13xi8, #hivm.address_space<ub>> -> !d_byte
    "tla.vec.func"() ({
      %ss = "tla.make_shape"() : () -> !tla.shape<256>
      %ds = "tla.make_shape"() : () -> !tla.shape<13>
      %coord = "tla.make_coord"() : () -> !tla.coord<0>
      %st = "tla.tile_view"(%s, %ss, %coord) : (!s_byte, !tla.shape<256>, !tla.coord<0>) -> !s_byte
      %dt = "tla.tile_view"(%d, %ds, %coord) : (!d_byte, !tla.shape<13>, !tla.coord<0>) -> !d_byte
      %v = tla.load %st : !s_byte -> !tla.vector<256xi8>
      tla.store %dt, %v {store_dist = #tla.store_dist<pack4_b32>} : !d_byte, !tla.vector<256xi8>
    }) : () -> ()
    return
  }

// CHECK-LABEL: func.func @pack4_half
// CHECK-LABEL: func.func private @vector_region_
// CHECK: %[[N_HALF:.*]] = arith.constant 26 : index
// CHECK: ave.hir.plt %[[N_HALF]]
// CHECK: ave.hir.masked_store <PK4_B32>
// CHECK-SAME: vector<128xi16>
  func.func @pack4_half(%sm: memref<128xi16, #hivm.address_space<ub>>, %dm: memref<13xi8, #hivm.address_space<ub>>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %cs = arith.constant 128 : index
    %cd = arith.constant 13 : index
    %s = tla.tensor_desc %sm shape [%c1, %cs, %c1, %c1] stride [%cs, %c1, %c1, %c1] origin_shape [%c1, %cs] coord [%c0, %c0] : memref<128xi16, #hivm.address_space<ub>> -> !s_half
    %d = tla.tensor_desc %dm shape [%c1, %cd, %c1, %c1] stride [%cd, %c1, %c1, %c1] origin_shape [%c1, %cd] coord [%c0, %c0] : memref<13xi8, #hivm.address_space<ub>> -> !d_half
    "tla.vec.func"() ({
      %ss = "tla.make_shape"() : () -> !tla.shape<128>
      %ds = "tla.make_shape"() : () -> !tla.shape<13>
      %coord = "tla.make_coord"() : () -> !tla.coord<0>
      %st = "tla.tile_view"(%s, %ss, %coord) : (!s_half, !tla.shape<128>, !tla.coord<0>) -> !s_half
      %dt = "tla.tile_view"(%d, %ds, %coord) : (!d_half, !tla.shape<13>, !tla.coord<0>) -> !d_half
      %v = tla.load %st : !s_half -> !tla.vector<128xi16>
      tla.store %dt, %v {store_dist = #tla.store_dist<pack4_b32>} : !d_half, !tla.vector<128xi16>
    }) : () -> ()
    return
  }

// CHECK-LABEL: func.func @pack4_word
// CHECK-LABEL: func.func private @vector_region_
// CHECK: %[[N_WORD:.*]] = arith.constant 13 : index
// CHECK: ave.hir.plt %[[N_WORD]]
// CHECK: ave.hir.masked_store <PK4_B32>
// CHECK-SAME: vector<64xi32>
  func.func @pack4_word(%sm: memref<64xi32, #hivm.address_space<ub>>, %dm: memref<13xi8, #hivm.address_space<ub>>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %cs = arith.constant 64 : index
    %cd = arith.constant 13 : index
    %s = tla.tensor_desc %sm shape [%c1, %cs, %c1, %c1] stride [%cs, %c1, %c1, %c1] origin_shape [%c1, %cs] coord [%c0, %c0] : memref<64xi32, #hivm.address_space<ub>> -> !s_word
    %d = tla.tensor_desc %dm shape [%c1, %cd, %c1, %c1] stride [%cd, %c1, %c1, %c1] origin_shape [%c1, %cd] coord [%c0, %c0] : memref<13xi8, #hivm.address_space<ub>> -> !d_word
    "tla.vec.func"() ({
      %ss = "tla.make_shape"() : () -> !tla.shape<64>
      %ds = "tla.make_shape"() : () -> !tla.shape<13>
      %coord = "tla.make_coord"() : () -> !tla.coord<0>
      %st = "tla.tile_view"(%s, %ss, %coord) : (!s_word, !tla.shape<64>, !tla.coord<0>) -> !s_word
      %dt = "tla.tile_view"(%d, %ds, %coord) : (!d_word, !tla.shape<13>, !tla.coord<0>) -> !d_word
      %v = tla.load %st : !s_word -> !tla.vector<64xi32>
      tla.store %dt, %v {store_dist = #tla.store_dist<pack4_b32>} : !d_word, !tla.vector<64xi32>
    }) : () -> ()
    return
  }

// CHECK-LABEL: func.func @pack4_full
// CHECK-LABEL: func.func private @vector_region_
// CHECK-NOT: ave.hir.plt
// CHECK: ave.hir.masked_store <PK4_B32>
// CHECK-SAME: vector<256xf8E5M2>
  func.func @pack4_full(%sm: memref<256xf8E5M2, #hivm.address_space<ub>>, %dm: memref<64xi8, #hivm.address_space<ub>>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %cs = arith.constant 256 : index
    %cd = arith.constant 64 : index
    %s = tla.tensor_desc %sm shape [%c1, %cs, %c1, %c1] stride [%cs, %c1, %c1, %c1] origin_shape [%c1, %cs] coord [%c0, %c0] : memref<256xf8E5M2, #hivm.address_space<ub>> -> !s_full
    %d = tla.tensor_desc %dm shape [%c1, %cd, %c1, %c1] stride [%cd, %c1, %c1, %c1] origin_shape [%c1, %cd] coord [%c0, %c0] : memref<64xi8, #hivm.address_space<ub>> -> !d_full
    "tla.vec.func"() ({
      %ss = "tla.make_shape"() : () -> !tla.shape<256>
      %ds = "tla.make_shape"() : () -> !tla.shape<64>
      %coord = "tla.make_coord"() : () -> !tla.coord<0>
      %st = "tla.tile_view"(%s, %ss, %coord) : (!s_full, !tla.shape<256>, !tla.coord<0>) -> !s_full
      %dt = "tla.tile_view"(%d, %ds, %coord) : (!d_full, !tla.shape<64>, !tla.coord<0>) -> !d_full
      %v = tla.load %st : !s_full -> !tla.vector<256xf8E5M2>
      tla.store %dt, %v {store_dist = #tla.store_dist<pack4_b32>} : !d_full, !tla.vector<256xf8E5M2>
    }) : () -> ()
    return
  }

// CHECK-LABEL: func.func @unpack4_e4m3
// CHECK-LABEL: func.func private @vector_region_
// CHECK: ave.hir.vload <UNPK4_B8>
// CHECK-SAME: vector<256xf8E4M3FN>
// CHECK: ave.hir.masked_store <NORM_B8>
  func.func @unpack4_e4m3(%sm: memref<64xf8E4M3FN, #hivm.address_space<ub>>, %dm: memref<256xf8E4M3FN, #hivm.address_space<ub>>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c64 = arith.constant 64 : index
    %c256 = arith.constant 256 : index
    %s = tla.tensor_desc %sm shape [%c1, %c64, %c1, %c1] stride [%c64, %c1, %c1, %c1] origin_shape [%c1, %c64] coord [%c0, %c0] : memref<64xf8E4M3FN, #hivm.address_space<ub>> -> !us_e4m3
    %d = tla.tensor_desc %dm shape [%c1, %c256, %c1, %c1] stride [%c256, %c1, %c1, %c1] origin_shape [%c1, %c256] coord [%c0, %c0] : memref<256xf8E4M3FN, #hivm.address_space<ub>> -> !ud_e4m3
    "tla.vec.func"() ({
      %ss = "tla.make_shape"() : () -> !tla.shape<64>
      %ds = "tla.make_shape"() : () -> !tla.shape<256>
      %coord = "tla.make_coord"() : () -> !tla.coord<0>
      %st = "tla.tile_view"(%s, %ss, %coord) : (!us_e4m3, !tla.shape<64>, !tla.coord<0>) -> !us_e4m3
      %dt = "tla.tile_view"(%d, %ds, %coord) : (!ud_e4m3, !tla.shape<256>, !tla.coord<0>) -> !ud_e4m3
      %v = tla.load %st {load_dist = #tla.load_dist<unpack4_b8>} : !us_e4m3 -> !tla.vector<256xf8E4M3FN>
      tla.store %dt, %v : !ud_e4m3, !tla.vector<256xf8E4M3FN>
    }) : () -> ()
    return
  }
}
