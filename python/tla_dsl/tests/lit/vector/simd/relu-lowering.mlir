// RUN: %tla_compile %s -o %t
// RUN: %filecheck %s --implicit-check-not=relu_int64_t --implicit-check-not=_mlir_ciface_vmax --implicit-check-not=hivm.vmul --implicit-check-not=hivm.vmax --implicit-check-not=hivm.pand --implicit-check-not=hivm.por --implicit-check-not=llvm.hivm.psel < %t

!tile_f16 = !tla.tensor<!tla.layout<!tla.shape<128>, !tla.stride<1>, !tla.shape<128>, RowMajor>, !tla.coord<0>, !tla.ptr<f16, ub, 4>>
!reg_f16 = !tla.vector<128xf16>
!tile_f32 = !tla.tensor<!tla.layout<!tla.shape<64>, !tla.stride<1>, !tla.shape<64>, RowMajor>, !tla.coord<0>, !tla.ptr<f32, ub, 4>>
!reg_f32 = !tla.vector<64xf32>

!tile_i32 = !tla.tensor<!tla.layout<!tla.shape<64>, !tla.stride<1>, !tla.shape<64>, RowMajor>, !tla.coord<0>, !tla.ptr<i32, ub, 4>>
!reg_i32 = !tla.vector<64xi32>
!tile_i64 = !tla.tensor<!tla.layout<!tla.shape<32>, !tla.stride<1>, !tla.shape<32>, RowMajor>, !tla.coord<0>, !tla.ptr<i64, ub, 8>>
!reg_i64 = !tla.vector<32xi64>

module {
  func.func @relu_f16(%input: memref<128xf16, #hivm.address_space<ub>>, %slope: f16) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %clanes = arith.constant 128 : index
    %tile = tla.tensor_desc %input shape [%c1, %clanes, %c1, %c1] stride [%clanes, %c1, %c1, %c1] origin_shape [%c1, %clanes] coord [%c0, %c0] : memref<128xf16, #hivm.address_space<ub>> -> !tile_f16
    "tla.vector"() ({
      "tla.vec.func"() ({
        %x = tla.load %tile : !tile_f16 -> !reg_f16
        %slopes = "tla.full"(%slope) : (f16) -> !reg_f16
        %mask = "tla.create_mask"() {pattern = "VL16", dtype = f16} : () -> !tla.mask<128>
        %r = tla.relu %x : !reg_f16 -> !reg_f16
        %p = tla.prelu %r, %slopes mask %mask : !reg_f16, !reg_f16 mask !tla.mask<128> -> !reg_f16
        %y = tla.leaky_relu %p, %slope : !reg_f16, f16 -> !reg_f16
        tla.store %tile, %y : !tile_f16, !reg_f16
      }) : () -> ()
    }) : () -> ()
    return
  }
  func.func @relu_f32(%input: memref<64xf32, #hivm.address_space<ub>>, %slope: f32) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %clanes = arith.constant 64 : index
    %tile = tla.tensor_desc %input shape [%c1, %clanes, %c1, %c1] stride [%clanes, %c1, %c1, %c1] origin_shape [%c1, %clanes] coord [%c0, %c0] : memref<64xf32, #hivm.address_space<ub>> -> !tile_f32
    "tla.vector"() ({
      "tla.vec.func"() ({
        %x = tla.load %tile : !tile_f32 -> !reg_f32
        %slopes = "tla.full"(%slope) : (f32) -> !reg_f32
        %mask = "tla.create_mask"() {pattern = "VL16", dtype = f32} : () -> !tla.mask<64>
        %r = tla.relu %x : !reg_f32 -> !reg_f32
        %p = tla.prelu %r, %slopes mask %mask : !reg_f32, !reg_f32 mask !tla.mask<64> -> !reg_f32
        %y = tla.leaky_relu %p, %slope : !reg_f32, f32 -> !reg_f32
        tla.store %tile, %y : !tile_f32, !reg_f32
      }) : () -> ()
    }) : () -> ()
    return
  }
  func.func @relu_i32(%input: memref<64xi32, #hivm.address_space<ub>>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %clanes = arith.constant 64 : index
    %tile = tla.tensor_desc %input shape [%c1, %clanes, %c1, %c1] stride [%clanes, %c1, %c1, %c1] origin_shape [%c1, %clanes] coord [%c0, %c0] : memref<64xi32, #hivm.address_space<ub>> -> !tile_i32
    "tla.vector"() ({
      "tla.vec.func"() ({
        %x = tla.load %tile : !tile_i32 -> !reg_i32
        %mask = "tla.create_mask"() {pattern = "VL16", dtype = i32} : () -> !tla.mask<64>
        %y = tla.relu %x mask %mask : !reg_i32 mask !tla.mask<64> -> !reg_i32
        tla.store %tile, %y : !tile_i32, !reg_i32
      }) : () -> ()
    }) : () -> ()
    return
  }
  func.func @relu_i64(%input: memref<32xi64, #hivm.address_space<ub>>, %threshold: i64) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %clanes = arith.constant 32 : index
    %tile = tla.tensor_desc %input shape [%c1, %clanes, %c1, %c1] stride [%clanes, %c1, %c1, %c1] origin_shape [%c1, %clanes] coord [%c0, %c0] : memref<32xi64, #hivm.address_space<ub>> -> !tile_i64
    "tla.vector"() ({
      "tla.vec.func"() ({
        %x = tla.load %tile : !tile_i64 -> !reg_i64
        %mask = "tla.create_mask"() {pattern = "VL16", dtype = i64} : () -> !tla.mask<32>
        %relu = tla.relu %x mask %mask : !reg_i64 mask !tla.mask<32> -> !reg_i64
        %max = tla.max %x, %relu mask %mask : !reg_i64, !reg_i64 mask !tla.mask<32> -> !reg_i64
        %y = tla.maxs %max, %threshold : !reg_i64, i64 -> !reg_i64
        tla.store %tile, %y : !tile_i64, !reg_i64
      }) : () -> ()
    }) : () -> ()
    return
  }
}

// CHECK-LABEL: func.func private @vector_region_
// CHECK: hivm.vrelu.x
// CHECK-SAME: vector<128xf16>
// CHECK: %[[PRELU_F16:.*]] = {{.*}}hivm.vprelu.x
// CHECK-SAME: vector<128xf16>
// CHECK: hivm.vlrelu.x{{.*}}%[[PRELU_F16]]
// CHECK-SAME: (vector<128xf16>, f16, vector<256xi1>)

// CHECK-LABEL: func.func private @vector_region_
// CHECK: hivm.vrelu.x
// CHECK-SAME: vector<64xf32>
// CHECK: %[[PRELU_F32:.*]] = {{.*}}hivm.vprelu.x
// CHECK-SAME: vector<64xf32>
// CHECK: hivm.vlrelu.x{{.*}}%[[PRELU_F32]]
// CHECK-SAME: (vector<64xf32>, f32, vector<256xi1>)

// CHECK-LABEL: func.func private @vector_region_
// CHECK: hivm.vrelu.x
// CHECK-SAME: vector<64xi32>

// CHECK-LABEL: func.func private @vector_region_
// CHECK-NOT: hivm.vrelu
// CHECK: hivm.ppack
// CHECK: hivm.vcmp.gt.s.z
// CHECK: hivm.vcmp.eq.s.z
// CHECK: hivm.vcmp.gt.u.z
// CHECK: call @tla_mask_select
// CHECK: hivm.vsel
// CHECK: hivm.vintlv
// CHECK: hivm.vcmp.gt.s.z
// CHECK: hivm.vcmp.gt.u.z
// CHECK: call @tla_mask_select
// CHECK: hivm.vsel
// CHECK: hivm.vintlv
// CHECK: hivm.vcmp.gt.s.z
// CHECK: hivm.vcmp.gt.u.z
// CHECK: call @tla_mask_select
// CHECK: hivm.vsel
// CHECK: hivm.vintlv
