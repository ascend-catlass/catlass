// RUN: %tla_compile %s -o %t
// RUN: %filecheck %s --check-prefix=CHECK < %t
// RUN: %filecheck %s --check-prefix=F16 < %t
// RUN: %filecheck %s --check-prefix=F32 < %t
//
// A fill region whose coord/shape leaves are runtime values (the mx_matmul pad
// region is exactly this: k_valid is a runtime chunk extent). The dynamic
// leaves travel as index operands of the make ops, get cast to i64, and land
// at the tail of the runtime call after the static descriptor fields.

module {
  tla.func @fill_l1_dyn_region(%crd1: index, %ext1: index) {
    %raw = tla.alloc_ptr{size_bytes = 4096} -> !tla.ptr<i8, l1, 512>
    %ptr = tla.recast_ptr %raw : !tla.ptr<i8, l1, 512> -> !tla.ptr<f8E4M3FN, l1, 512>
    %shape = tla.make_shape -> !tla.shape<(16,2),(32,2)>
    %stride = tla.make_stride -> !tla.stride<(32,512),(1,1024)>
    %origin = tla.make_shape -> !tla.shape<32,64>
    %layout = tla.make_layout %shape, %stride origin %origin {layoutTag = #tla.layout_tag<zN>} :
      !tla.shape<(16,2),(32,2)>, !tla.stride<(32,512),(1,1024)> origin !tla.shape<32,64> ->
      !tla.layout<!tla.shape<(16,2),(32,2)>, !tla.stride<(32,512),(1,1024)>, !tla.shape<32,64>, zN>
    %zero_coord = tla.make_coord -> !tla.coord<0,0>
    %dst = tla.make_tensor %ptr, %layout, %zero_coord :
      !tla.ptr<f8E4M3FN, l1, 512>,
      !tla.layout<!tla.shape<(16,2),(32,2)>, !tla.stride<(32,512),(1,1024)>, !tla.shape<32,64>, zN>,
      !tla.coord<0,0> ->
      !tla.tensor<!tla.layout<!tla.shape<(16,2),(32,2)>, !tla.stride<(32,512),(1,1024)>, !tla.shape<32,64>, zN>, !tla.coord<0,0>, !tla.ptr<f8E4M3FN, l1, 512>>
    %zero = arith.constant 0 : i32
    %fill_coord = tla.make_coord %crd1 -> !tla.coord<0,?>
    %fill_shape = tla.make_shape %ext1 -> !tla.shape<32,?>
    "tla.cube"() ({
      tla.fill %dst, %zero, %fill_coord, %fill_shape :
        !tla.tensor<!tla.layout<!tla.shape<(16,2),(32,2)>, !tla.stride<(32,512),(1,1024)>, !tla.shape<32,64>, zN>, !tla.coord<0,0>, !tla.ptr<f8E4M3FN, l1, 512>>,
        i32, !tla.coord<0,?>, !tla.shape<32,?>
    }) : () -> ()
    tla.return
  }

  tla.func @fill_l1_dyn_region_f16(%crd1: index, %ext1: index) {
    %raw = tla.alloc_ptr{size_bytes = 4096} -> !tla.ptr<i8, l1, 512>
    %ptr = tla.recast_ptr %raw : !tla.ptr<i8, l1, 512> -> !tla.ptr<f16, l1, 512>
    %shape = tla.make_shape -> !tla.shape<(16,2),(16,4)>
    %stride = tla.make_stride -> !tla.stride<(16,256),(1,512)>
    %origin = tla.make_shape -> !tla.shape<32,64>
    %layout = tla.make_layout %shape, %stride origin %origin {layoutTag = #tla.layout_tag<zN>} :
      !tla.shape<(16,2),(16,4)>, !tla.stride<(16,256),(1,512)> origin !tla.shape<32,64> ->
      !tla.layout<!tla.shape<(16,2),(16,4)>, !tla.stride<(16,256),(1,512)>, !tla.shape<32,64>, zN>
    %zero_coord = tla.make_coord -> !tla.coord<0,0>
    %dst = tla.make_tensor %ptr, %layout, %zero_coord :
      !tla.ptr<f16, l1, 512>,
      !tla.layout<!tla.shape<(16,2),(16,4)>, !tla.stride<(16,256),(1,512)>, !tla.shape<32,64>, zN>,
      !tla.coord<0,0> ->
      !tla.tensor<!tla.layout<!tla.shape<(16,2),(16,4)>, !tla.stride<(16,256),(1,512)>, !tla.shape<32,64>, zN>, !tla.coord<0,0>, !tla.ptr<f16, l1, 512>>
    %zero = arith.constant 0 : i32
    %fill_coord = tla.make_coord %crd1 -> !tla.coord<0,?>
    %fill_shape = tla.make_shape %ext1 -> !tla.shape<32,?>
    "tla.cube"() ({
      tla.fill %dst, %zero, %fill_coord, %fill_shape :
        !tla.tensor<!tla.layout<!tla.shape<(16,2),(16,4)>, !tla.stride<(16,256),(1,512)>, !tla.shape<32,64>, zN>, !tla.coord<0,0>, !tla.ptr<f16, l1, 512>>,
        i32, !tla.coord<0,?>, !tla.shape<32,?>
    }) : () -> ()
    tla.return
  }

  tla.func @fill_l1_dyn_region_f32(%crd1: index, %ext1: index) {
    %raw = tla.alloc_ptr{size_bytes = 4096} -> !tla.ptr<i8, l1, 512>
    %ptr = tla.recast_ptr %raw : !tla.ptr<i8, l1, 512> -> !tla.ptr<f32, l1, 512>
    %shape = tla.make_shape -> !tla.shape<(16,2),(8,8)>
    %stride = tla.make_stride -> !tla.stride<(8,128),(1,256)>
    %origin = tla.make_shape -> !tla.shape<32,64>
    %layout = tla.make_layout %shape, %stride origin %origin {layoutTag = #tla.layout_tag<zN>} :
      !tla.shape<(16,2),(8,8)>, !tla.stride<(8,128),(1,256)> origin !tla.shape<32,64> ->
      !tla.layout<!tla.shape<(16,2),(8,8)>, !tla.stride<(8,128),(1,256)>, !tla.shape<32,64>, zN>
    %zero_coord = tla.make_coord -> !tla.coord<0,0>
    %dst = tla.make_tensor %ptr, %layout, %zero_coord :
      !tla.ptr<f32, l1, 512>,
      !tla.layout<!tla.shape<(16,2),(8,8)>, !tla.stride<(8,128),(1,256)>, !tla.shape<32,64>, zN>,
      !tla.coord<0,0> ->
      !tla.tensor<!tla.layout<!tla.shape<(16,2),(8,8)>, !tla.stride<(8,128),(1,256)>, !tla.shape<32,64>, zN>, !tla.coord<0,0>, !tla.ptr<f32, l1, 512>>
    %zero = arith.constant 0 : i32
    %fill_coord = tla.make_coord %crd1 -> !tla.coord<0,?>
    %fill_shape = tla.make_shape %ext1 -> !tla.shape<32,?>
    "tla.cube"() ({
      tla.fill %dst, %zero, %fill_coord, %fill_shape :
        !tla.tensor<!tla.layout<!tla.shape<(16,2),(8,8)>, !tla.stride<(8,128),(1,256)>, !tla.shape<32,64>, zN>, !tla.coord<0,0>, !tla.ptr<f32, l1, 512>>,
        i32, !tla.coord<0,?>, !tla.shape<32,?>
    }) : () -> ()
    tla.return
  }
}

// CHECK: func.func private @fill_l1_zN_fp8_e4m3fn_t(
// CHECK-LABEL: func.func @fill_l1_dyn_region(%arg0: index, %arg1: index)
// The dynamic coord/shape leaves reach the call as the i64 casts of the entry
// arguments, after the static region leaves (ext0 = 32 sits between them).
// CHECK-DAG: %[[CRD1:.+]] = builtin.unrealized_conversion_cast %arg0 : index to i64
// CHECK-DAG: %[[EXT1:.+]] = builtin.unrealized_conversion_cast %arg1 : index to i64
// CHECK: call @fill_l1_zN_fp8_e4m3fn_t
// CHECK-SAME: %[[CRD1]], {{%[0-9]+}}, %[[EXT1]], {{%[0-9]+}})
// CHECK-NOT: tla.fill

// F16: func.func private @fill_l1_zN_half(
// F16-LABEL: func.func @fill_l1_dyn_region_f16(%arg0: index, %arg1: index)
// F16-DAG: %[[CRD1:.+]] = builtin.unrealized_conversion_cast %arg0 : index to i64
// F16-DAG: %[[EXT1:.+]] = builtin.unrealized_conversion_cast %arg1 : index to i64
// F16: call @fill_l1_zN_half
// F16-SAME: %[[CRD1]], {{%[0-9]+}}, %[[EXT1]], {{%[0-9]+}})
// F16-NOT: tla.fill

// F32: func.func private @fill_l1_zN_float(
// F32-LABEL: func.func @fill_l1_dyn_region_f32(%arg0: index, %arg1: index)
// F32-DAG: %[[CRD1:.+]] = builtin.unrealized_conversion_cast %arg0 : index to i64
// F32-DAG: %[[EXT1:.+]] = builtin.unrealized_conversion_cast %arg1 : index to i64
// F32: call @fill_l1_zN_float
// F32-SAME: %[[CRD1]], {{%[0-9]+}}, %[[EXT1]], {{%[0-9]+}})
// F32-NOT: tla.fill
