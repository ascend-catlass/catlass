// RUN: %tla_compile %s --mlir-print-ir-after=tla-vector-region-simd -o %t 2>&1 | %filecheck %s --implicit-check-not=tla.pack --implicit-check-not=tla.unpack --implicit-check-not=ave.hir.vpack --implicit-check-not=ave.hir.vunpack --implicit-check-not=intr.hivm.vpack --implicit-check-not=intr.hivm.vzunpack --implicit-check-not=intr.hivm.vsunpack

// The signedness in each symbol selects the CANN template specialization.
// The call ABI uses signless vector carriers, including unsigned inputs/results.

!t8 = !tla.tensor<!tla.layout<!tla.shape<256>, !tla.stride<1>, !tla.shape<256>, RowMajor>, !tla.coord<0>, !tla.ptr<i8, ub, 1>>
!t16 = !tla.tensor<!tla.layout<!tla.shape<128>, !tla.stride<1>, !tla.shape<128>, RowMajor>, !tla.coord<0>, !tla.ptr<i16, ub, 2>>
!t32 = !tla.tensor<!tla.layout<!tla.shape<64>, !tla.stride<1>, !tla.shape<64>, RowMajor>, !tla.coord<0>, !tla.ptr<i32, ub, 4>>

module {
  func.func @pack_i16_u8_low(%src_mem: memref<128xi16, #hivm.address_space<ub>>, %dst_mem: memref<256xi8, #hivm.address_space<ub>>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %sn = arith.constant 128 : index
    %dn = arith.constant 256 : index
    %src = tla.tensor_desc %src_mem shape [%c1, %sn, %c1, %c1] stride [%sn, %c1, %c1, %c1] origin_shape [%c1, %sn] coord [%c0, %c0] : memref<128xi16, #hivm.address_space<ub>> -> !t16
    %dst = tla.tensor_desc %dst_mem shape [%c1, %dn, %c1, %c1] stride [%dn, %c1, %c1, %c1] origin_shape [%c1, %dn] coord [%c0, %c0] : memref<256xi8, #hivm.address_space<ub>> -> !t8
    "tla.vec.func"() ({
      %loaded = tla.load %src : !t16 -> !tla.vector<128xi16>
      %result = tla.pack %loaded, <low> : !tla.vector<128xi16> -> !tla.vector<256xui8>
      %signed = tla.bitcast %result : !tla.vector<256xui8> -> !tla.vector<256xi8>
      tla.store %dst, %signed : !t8, !tla.vector<256xi8>
    }) {mode = "simd"} : () -> ()
    return
  }

  func.func @pack_i16_u8_high(%src_mem: memref<128xi16, #hivm.address_space<ub>>, %dst_mem: memref<256xi8, #hivm.address_space<ub>>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %sn = arith.constant 128 : index
    %dn = arith.constant 256 : index
    %src = tla.tensor_desc %src_mem shape [%c1, %sn, %c1, %c1] stride [%sn, %c1, %c1, %c1] origin_shape [%c1, %sn] coord [%c0, %c0] : memref<128xi16, #hivm.address_space<ub>> -> !t16
    %dst = tla.tensor_desc %dst_mem shape [%c1, %dn, %c1, %c1] stride [%dn, %c1, %c1, %c1] origin_shape [%c1, %dn] coord [%c0, %c0] : memref<256xi8, #hivm.address_space<ub>> -> !t8
    "tla.vec.func"() ({
      %loaded = tla.load %src : !t16 -> !tla.vector<128xi16>
      %result = tla.pack %loaded, <high> : !tla.vector<128xi16> -> !tla.vector<256xui8>
      %signed = tla.bitcast %result : !tla.vector<256xui8> -> !tla.vector<256xi8>
      tla.store %dst, %signed : !t8, !tla.vector<256xi8>
    }) {mode = "simd"} : () -> ()
    return
  }

  func.func @pack_u16_u8_low(%src_mem: memref<128xi16, #hivm.address_space<ub>>, %dst_mem: memref<256xi8, #hivm.address_space<ub>>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %sn = arith.constant 128 : index
    %dn = arith.constant 256 : index
    %src = tla.tensor_desc %src_mem shape [%c1, %sn, %c1, %c1] stride [%sn, %c1, %c1, %c1] origin_shape [%c1, %sn] coord [%c0, %c0] : memref<128xi16, #hivm.address_space<ub>> -> !t16
    %dst = tla.tensor_desc %dst_mem shape [%c1, %dn, %c1, %c1] stride [%dn, %c1, %c1, %c1] origin_shape [%c1, %dn] coord [%c0, %c0] : memref<256xi8, #hivm.address_space<ub>> -> !t8
    "tla.vec.func"() ({
      %loaded = tla.load %src : !t16 -> !tla.vector<128xi16>
      %unsigned = tla.bitcast %loaded : !tla.vector<128xi16> -> !tla.vector<128xui16>
      %result = tla.pack %unsigned, <low> : !tla.vector<128xui16> -> !tla.vector<256xui8>
      %signed = tla.bitcast %result : !tla.vector<256xui8> -> !tla.vector<256xi8>
      tla.store %dst, %signed : !t8, !tla.vector<256xi8>
    }) {mode = "simd"} : () -> ()
    return
  }

  func.func @pack_u16_u8_high(%src_mem: memref<128xi16, #hivm.address_space<ub>>, %dst_mem: memref<256xi8, #hivm.address_space<ub>>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %sn = arith.constant 128 : index
    %dn = arith.constant 256 : index
    %src = tla.tensor_desc %src_mem shape [%c1, %sn, %c1, %c1] stride [%sn, %c1, %c1, %c1] origin_shape [%c1, %sn] coord [%c0, %c0] : memref<128xi16, #hivm.address_space<ub>> -> !t16
    %dst = tla.tensor_desc %dst_mem shape [%c1, %dn, %c1, %c1] stride [%dn, %c1, %c1, %c1] origin_shape [%c1, %dn] coord [%c0, %c0] : memref<256xi8, #hivm.address_space<ub>> -> !t8
    "tla.vec.func"() ({
      %loaded = tla.load %src : !t16 -> !tla.vector<128xi16>
      %unsigned = tla.bitcast %loaded : !tla.vector<128xi16> -> !tla.vector<128xui16>
      %result = tla.pack %unsigned, <high> : !tla.vector<128xui16> -> !tla.vector<256xui8>
      %signed = tla.bitcast %result : !tla.vector<256xui8> -> !tla.vector<256xi8>
      tla.store %dst, %signed : !t8, !tla.vector<256xi8>
    }) {mode = "simd"} : () -> ()
    return
  }

  func.func @pack_i32_u16_low(%src_mem: memref<64xi32, #hivm.address_space<ub>>, %dst_mem: memref<128xi16, #hivm.address_space<ub>>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %sn = arith.constant 64 : index
    %dn = arith.constant 128 : index
    %src = tla.tensor_desc %src_mem shape [%c1, %sn, %c1, %c1] stride [%sn, %c1, %c1, %c1] origin_shape [%c1, %sn] coord [%c0, %c0] : memref<64xi32, #hivm.address_space<ub>> -> !t32
    %dst = tla.tensor_desc %dst_mem shape [%c1, %dn, %c1, %c1] stride [%dn, %c1, %c1, %c1] origin_shape [%c1, %dn] coord [%c0, %c0] : memref<128xi16, #hivm.address_space<ub>> -> !t16
    "tla.vec.func"() ({
      %loaded = tla.load %src : !t32 -> !tla.vector<64xi32>
      %result = tla.pack %loaded, <low> : !tla.vector<64xi32> -> !tla.vector<128xui16>
      %signed = tla.bitcast %result : !tla.vector<128xui16> -> !tla.vector<128xi16>
      tla.store %dst, %signed : !t16, !tla.vector<128xi16>
    }) {mode = "simd"} : () -> ()
    return
  }

  func.func @pack_i32_u16_high(%src_mem: memref<64xi32, #hivm.address_space<ub>>, %dst_mem: memref<128xi16, #hivm.address_space<ub>>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %sn = arith.constant 64 : index
    %dn = arith.constant 128 : index
    %src = tla.tensor_desc %src_mem shape [%c1, %sn, %c1, %c1] stride [%sn, %c1, %c1, %c1] origin_shape [%c1, %sn] coord [%c0, %c0] : memref<64xi32, #hivm.address_space<ub>> -> !t32
    %dst = tla.tensor_desc %dst_mem shape [%c1, %dn, %c1, %c1] stride [%dn, %c1, %c1, %c1] origin_shape [%c1, %dn] coord [%c0, %c0] : memref<128xi16, #hivm.address_space<ub>> -> !t16
    "tla.vec.func"() ({
      %loaded = tla.load %src : !t32 -> !tla.vector<64xi32>
      %result = tla.pack %loaded, <high> : !tla.vector<64xi32> -> !tla.vector<128xui16>
      %signed = tla.bitcast %result : !tla.vector<128xui16> -> !tla.vector<128xi16>
      tla.store %dst, %signed : !t16, !tla.vector<128xi16>
    }) {mode = "simd"} : () -> ()
    return
  }

  func.func @pack_u32_u16_low(%src_mem: memref<64xi32, #hivm.address_space<ub>>, %dst_mem: memref<128xi16, #hivm.address_space<ub>>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %sn = arith.constant 64 : index
    %dn = arith.constant 128 : index
    %src = tla.tensor_desc %src_mem shape [%c1, %sn, %c1, %c1] stride [%sn, %c1, %c1, %c1] origin_shape [%c1, %sn] coord [%c0, %c0] : memref<64xi32, #hivm.address_space<ub>> -> !t32
    %dst = tla.tensor_desc %dst_mem shape [%c1, %dn, %c1, %c1] stride [%dn, %c1, %c1, %c1] origin_shape [%c1, %dn] coord [%c0, %c0] : memref<128xi16, #hivm.address_space<ub>> -> !t16
    "tla.vec.func"() ({
      %loaded = tla.load %src : !t32 -> !tla.vector<64xi32>
      %unsigned = tla.bitcast %loaded : !tla.vector<64xi32> -> !tla.vector<64xui32>
      %result = tla.pack %unsigned, <low> : !tla.vector<64xui32> -> !tla.vector<128xui16>
      %signed = tla.bitcast %result : !tla.vector<128xui16> -> !tla.vector<128xi16>
      tla.store %dst, %signed : !t16, !tla.vector<128xi16>
    }) {mode = "simd"} : () -> ()
    return
  }

  func.func @pack_u32_u16_high(%src_mem: memref<64xi32, #hivm.address_space<ub>>, %dst_mem: memref<128xi16, #hivm.address_space<ub>>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %sn = arith.constant 64 : index
    %dn = arith.constant 128 : index
    %src = tla.tensor_desc %src_mem shape [%c1, %sn, %c1, %c1] stride [%sn, %c1, %c1, %c1] origin_shape [%c1, %sn] coord [%c0, %c0] : memref<64xi32, #hivm.address_space<ub>> -> !t32
    %dst = tla.tensor_desc %dst_mem shape [%c1, %dn, %c1, %c1] stride [%dn, %c1, %c1, %c1] origin_shape [%c1, %dn] coord [%c0, %c0] : memref<128xi16, #hivm.address_space<ub>> -> !t16
    "tla.vec.func"() ({
      %loaded = tla.load %src : !t32 -> !tla.vector<64xi32>
      %unsigned = tla.bitcast %loaded : !tla.vector<64xi32> -> !tla.vector<64xui32>
      %result = tla.pack %unsigned, <high> : !tla.vector<64xui32> -> !tla.vector<128xui16>
      %signed = tla.bitcast %result : !tla.vector<128xui16> -> !tla.vector<128xi16>
      tla.store %dst, %signed : !t16, !tla.vector<128xi16>
    }) {mode = "simd"} : () -> ()
    return
  }

  func.func @unpack_i8_i16_low(%src_mem: memref<256xi8, #hivm.address_space<ub>>, %dst_mem: memref<128xi16, #hivm.address_space<ub>>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %sn = arith.constant 256 : index
    %dn = arith.constant 128 : index
    %src = tla.tensor_desc %src_mem shape [%c1, %sn, %c1, %c1] stride [%sn, %c1, %c1, %c1] origin_shape [%c1, %sn] coord [%c0, %c0] : memref<256xi8, #hivm.address_space<ub>> -> !t8
    %dst = tla.tensor_desc %dst_mem shape [%c1, %dn, %c1, %c1] stride [%dn, %c1, %c1, %c1] origin_shape [%c1, %dn] coord [%c0, %c0] : memref<128xi16, #hivm.address_space<ub>> -> !t16
    "tla.vec.func"() ({
      %loaded = tla.load %src : !t8 -> !tla.vector<256xi8>
      %result = tla.unpack %loaded, <low> : !tla.vector<256xi8> -> !tla.vector<128xi16>
      tla.store %dst, %result : !t16, !tla.vector<128xi16>
    }) {mode = "simd"} : () -> ()
    return
  }

  func.func @unpack_i8_i16_high(%src_mem: memref<256xi8, #hivm.address_space<ub>>, %dst_mem: memref<128xi16, #hivm.address_space<ub>>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %sn = arith.constant 256 : index
    %dn = arith.constant 128 : index
    %src = tla.tensor_desc %src_mem shape [%c1, %sn, %c1, %c1] stride [%sn, %c1, %c1, %c1] origin_shape [%c1, %sn] coord [%c0, %c0] : memref<256xi8, #hivm.address_space<ub>> -> !t8
    %dst = tla.tensor_desc %dst_mem shape [%c1, %dn, %c1, %c1] stride [%dn, %c1, %c1, %c1] origin_shape [%c1, %dn] coord [%c0, %c0] : memref<128xi16, #hivm.address_space<ub>> -> !t16
    "tla.vec.func"() ({
      %loaded = tla.load %src : !t8 -> !tla.vector<256xi8>
      %result = tla.unpack %loaded, <high> : !tla.vector<256xi8> -> !tla.vector<128xi16>
      tla.store %dst, %result : !t16, !tla.vector<128xi16>
    }) {mode = "simd"} : () -> ()
    return
  }

  func.func @unpack_u8_u16_low(%src_mem: memref<256xi8, #hivm.address_space<ub>>, %dst_mem: memref<128xi16, #hivm.address_space<ub>>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %sn = arith.constant 256 : index
    %dn = arith.constant 128 : index
    %src = tla.tensor_desc %src_mem shape [%c1, %sn, %c1, %c1] stride [%sn, %c1, %c1, %c1] origin_shape [%c1, %sn] coord [%c0, %c0] : memref<256xi8, #hivm.address_space<ub>> -> !t8
    %dst = tla.tensor_desc %dst_mem shape [%c1, %dn, %c1, %c1] stride [%dn, %c1, %c1, %c1] origin_shape [%c1, %dn] coord [%c0, %c0] : memref<128xi16, #hivm.address_space<ub>> -> !t16
    "tla.vec.func"() ({
      %loaded = tla.load %src : !t8 -> !tla.vector<256xi8>
      %unsigned = tla.bitcast %loaded : !tla.vector<256xi8> -> !tla.vector<256xui8>
      %result = tla.unpack %unsigned, <low> : !tla.vector<256xui8> -> !tla.vector<128xui16>
      %signed = tla.bitcast %result : !tla.vector<128xui16> -> !tla.vector<128xi16>
      tla.store %dst, %signed : !t16, !tla.vector<128xi16>
    }) {mode = "simd"} : () -> ()
    return
  }

  func.func @unpack_u8_u16_high(%src_mem: memref<256xi8, #hivm.address_space<ub>>, %dst_mem: memref<128xi16, #hivm.address_space<ub>>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %sn = arith.constant 256 : index
    %dn = arith.constant 128 : index
    %src = tla.tensor_desc %src_mem shape [%c1, %sn, %c1, %c1] stride [%sn, %c1, %c1, %c1] origin_shape [%c1, %sn] coord [%c0, %c0] : memref<256xi8, #hivm.address_space<ub>> -> !t8
    %dst = tla.tensor_desc %dst_mem shape [%c1, %dn, %c1, %c1] stride [%dn, %c1, %c1, %c1] origin_shape [%c1, %dn] coord [%c0, %c0] : memref<128xi16, #hivm.address_space<ub>> -> !t16
    "tla.vec.func"() ({
      %loaded = tla.load %src : !t8 -> !tla.vector<256xi8>
      %unsigned = tla.bitcast %loaded : !tla.vector<256xi8> -> !tla.vector<256xui8>
      %result = tla.unpack %unsigned, <high> : !tla.vector<256xui8> -> !tla.vector<128xui16>
      %signed = tla.bitcast %result : !tla.vector<128xui16> -> !tla.vector<128xi16>
      tla.store %dst, %signed : !t16, !tla.vector<128xi16>
    }) {mode = "simd"} : () -> ()
    return
  }

  func.func @unpack_i16_i32_low(%src_mem: memref<128xi16, #hivm.address_space<ub>>, %dst_mem: memref<64xi32, #hivm.address_space<ub>>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %sn = arith.constant 128 : index
    %dn = arith.constant 64 : index
    %src = tla.tensor_desc %src_mem shape [%c1, %sn, %c1, %c1] stride [%sn, %c1, %c1, %c1] origin_shape [%c1, %sn] coord [%c0, %c0] : memref<128xi16, #hivm.address_space<ub>> -> !t16
    %dst = tla.tensor_desc %dst_mem shape [%c1, %dn, %c1, %c1] stride [%dn, %c1, %c1, %c1] origin_shape [%c1, %dn] coord [%c0, %c0] : memref<64xi32, #hivm.address_space<ub>> -> !t32
    "tla.vec.func"() ({
      %loaded = tla.load %src : !t16 -> !tla.vector<128xi16>
      %result = tla.unpack %loaded, <low> : !tla.vector<128xi16> -> !tla.vector<64xi32>
      tla.store %dst, %result : !t32, !tla.vector<64xi32>
    }) {mode = "simd"} : () -> ()
    return
  }

  func.func @unpack_i16_i32_high(%src_mem: memref<128xi16, #hivm.address_space<ub>>, %dst_mem: memref<64xi32, #hivm.address_space<ub>>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %sn = arith.constant 128 : index
    %dn = arith.constant 64 : index
    %src = tla.tensor_desc %src_mem shape [%c1, %sn, %c1, %c1] stride [%sn, %c1, %c1, %c1] origin_shape [%c1, %sn] coord [%c0, %c0] : memref<128xi16, #hivm.address_space<ub>> -> !t16
    %dst = tla.tensor_desc %dst_mem shape [%c1, %dn, %c1, %c1] stride [%dn, %c1, %c1, %c1] origin_shape [%c1, %dn] coord [%c0, %c0] : memref<64xi32, #hivm.address_space<ub>> -> !t32
    "tla.vec.func"() ({
      %loaded = tla.load %src : !t16 -> !tla.vector<128xi16>
      %result = tla.unpack %loaded, <high> : !tla.vector<128xi16> -> !tla.vector<64xi32>
      tla.store %dst, %result : !t32, !tla.vector<64xi32>
    }) {mode = "simd"} : () -> ()
    return
  }

  func.func @unpack_u16_u32_low(%src_mem: memref<128xi16, #hivm.address_space<ub>>, %dst_mem: memref<64xi32, #hivm.address_space<ub>>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %sn = arith.constant 128 : index
    %dn = arith.constant 64 : index
    %src = tla.tensor_desc %src_mem shape [%c1, %sn, %c1, %c1] stride [%sn, %c1, %c1, %c1] origin_shape [%c1, %sn] coord [%c0, %c0] : memref<128xi16, #hivm.address_space<ub>> -> !t16
    %dst = tla.tensor_desc %dst_mem shape [%c1, %dn, %c1, %c1] stride [%dn, %c1, %c1, %c1] origin_shape [%c1, %dn] coord [%c0, %c0] : memref<64xi32, #hivm.address_space<ub>> -> !t32
    "tla.vec.func"() ({
      %loaded = tla.load %src : !t16 -> !tla.vector<128xi16>
      %unsigned = tla.bitcast %loaded : !tla.vector<128xi16> -> !tla.vector<128xui16>
      %result = tla.unpack %unsigned, <low> : !tla.vector<128xui16> -> !tla.vector<64xui32>
      %signed = tla.bitcast %result : !tla.vector<64xui32> -> !tla.vector<64xi32>
      tla.store %dst, %signed : !t32, !tla.vector<64xi32>
    }) {mode = "simd"} : () -> ()
    return
  }

  func.func @unpack_u16_u32_high(%src_mem: memref<128xi16, #hivm.address_space<ub>>, %dst_mem: memref<64xi32, #hivm.address_space<ub>>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %sn = arith.constant 128 : index
    %dn = arith.constant 64 : index
    %src = tla.tensor_desc %src_mem shape [%c1, %sn, %c1, %c1] stride [%sn, %c1, %c1, %c1] origin_shape [%c1, %sn] coord [%c0, %c0] : memref<128xi16, #hivm.address_space<ub>> -> !t16
    %dst = tla.tensor_desc %dst_mem shape [%c1, %dn, %c1, %c1] stride [%dn, %c1, %c1, %c1] origin_shape [%c1, %dn] coord [%c0, %c0] : memref<64xi32, #hivm.address_space<ub>> -> !t32
    "tla.vec.func"() ({
      %loaded = tla.load %src : !t16 -> !tla.vector<128xi16>
      %unsigned = tla.bitcast %loaded : !tla.vector<128xi16> -> !tla.vector<128xui16>
      %result = tla.unpack %unsigned, <high> : !tla.vector<128xui16> -> !tla.vector<64xui32>
      %signed = tla.bitcast %result : !tla.vector<64xui32> -> !tla.vector<64xi32>
      tla.store %dst, %signed : !t32, !tla.vector<64xi32>
    }) {mode = "simd"} : () -> ()
    return
  }

}

// CHECK-DAG: func.func private @pack_reg_uint8_t_int16_t_low(vector<128xi16>) -> vector<256xi8>
// CHECK-DAG: func.func private @pack_reg_uint8_t_int16_t_high(vector<128xi16>) -> vector<256xi8>
// CHECK-DAG: func.func private @pack_reg_uint8_t_uint16_t_low(vector<128xi16>) -> vector<256xi8>
// CHECK-DAG: func.func private @pack_reg_uint8_t_uint16_t_high(vector<128xi16>) -> vector<256xi8>
// CHECK-DAG: func.func private @pack_reg_uint16_t_int32_t_low(vector<64xi32>) -> vector<128xi16>
// CHECK-DAG: func.func private @pack_reg_uint16_t_int32_t_high(vector<64xi32>) -> vector<128xi16>
// CHECK-DAG: func.func private @pack_reg_uint16_t_uint32_t_low(vector<64xi32>) -> vector<128xi16>
// CHECK-DAG: func.func private @pack_reg_uint16_t_uint32_t_high(vector<64xi32>) -> vector<128xi16>
// CHECK-DAG: func.func private @unpack_reg_int16_t_int8_t_low(vector<256xi8>) -> vector<128xi16>
// CHECK-DAG: func.func private @unpack_reg_int16_t_int8_t_high(vector<256xi8>) -> vector<128xi16>
// CHECK-DAG: func.func private @unpack_reg_uint16_t_uint8_t_low(vector<256xi8>) -> vector<128xi16>
// CHECK-DAG: func.func private @unpack_reg_uint16_t_uint8_t_high(vector<256xi8>) -> vector<128xi16>
// CHECK-DAG: func.func private @unpack_reg_int32_t_int16_t_low(vector<128xi16>) -> vector<64xi32>
// CHECK-DAG: func.func private @unpack_reg_int32_t_int16_t_high(vector<128xi16>) -> vector<64xi32>
// CHECK-DAG: func.func private @unpack_reg_uint32_t_uint16_t_low(vector<128xi16>) -> vector<64xi32>
// CHECK-DAG: func.func private @unpack_reg_uint32_t_uint16_t_high(vector<128xi16>) -> vector<64xi32>

// CHECK: call @pack_reg_uint8_t_int16_t_low({{.*}}) : (vector<128xi16>) -> vector<256xi8>
// CHECK: call @pack_reg_uint8_t_int16_t_high({{.*}}) : (vector<128xi16>) -> vector<256xi8>
// CHECK: call @pack_reg_uint8_t_uint16_t_low({{.*}}) : (vector<128xi16>) -> vector<256xi8>
// CHECK: call @pack_reg_uint8_t_uint16_t_high({{.*}}) : (vector<128xi16>) -> vector<256xi8>
// CHECK: call @pack_reg_uint16_t_int32_t_low({{.*}}) : (vector<64xi32>) -> vector<128xi16>
// CHECK: call @pack_reg_uint16_t_int32_t_high({{.*}}) : (vector<64xi32>) -> vector<128xi16>
// CHECK: call @pack_reg_uint16_t_uint32_t_low({{.*}}) : (vector<64xi32>) -> vector<128xi16>
// CHECK: call @pack_reg_uint16_t_uint32_t_high({{.*}}) : (vector<64xi32>) -> vector<128xi16>
// CHECK: call @unpack_reg_int16_t_int8_t_low({{.*}}) : (vector<256xi8>) -> vector<128xi16>
// CHECK: call @unpack_reg_int16_t_int8_t_high({{.*}}) : (vector<256xi8>) -> vector<128xi16>
// CHECK: call @unpack_reg_uint16_t_uint8_t_low({{.*}}) : (vector<256xi8>) -> vector<128xi16>
// CHECK: call @unpack_reg_uint16_t_uint8_t_high({{.*}}) : (vector<256xi8>) -> vector<128xi16>
// CHECK: call @unpack_reg_int32_t_int16_t_low({{.*}}) : (vector<128xi16>) -> vector<64xi32>
// CHECK: call @unpack_reg_int32_t_int16_t_high({{.*}}) : (vector<128xi16>) -> vector<64xi32>
// CHECK: call @unpack_reg_uint32_t_uint16_t_low({{.*}}) : (vector<128xi16>) -> vector<64xi32>
// CHECK: call @unpack_reg_uint32_t_uint16_t_high({{.*}}) : (vector<128xi16>) -> vector<64xi32>
