// RUN: %tla_compile %s -o - | %filecheck %s

!tile = !tla.tensor<!tla.layout<!tla.shape<64>, !tla.stride<1>, !tla.shape<64>, RowMajor>, !tla.coord<0>, !tla.ptr<f32, ub, 32>>
func.func @inplace_mul_add_test(%dst_mem: memref<64xf32, #hivm.address_space<ub>>, %factor_mem: memref<64xf32, #hivm.address_space<ub>>, %bias_mem: memref<64xf32, #hivm.address_space<ub>>, %out_mem: memref<64xf32, #hivm.address_space<ub>>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %cn = arith.constant 64 : index
  %dst = tla.tensor_desc %dst_mem shape [%c1, %cn, %c1, %c1] stride [%cn, %c1, %c1, %c1] origin_shape [%c1, %cn] coord [%c0, %c0] : memref<64xf32, #hivm.address_space<ub>> -> !tile
  %factor = tla.tensor_desc %factor_mem shape [%c1, %cn, %c1, %c1] stride [%cn, %c1, %c1, %c1] origin_shape [%c1, %cn] coord [%c0, %c0] : memref<64xf32, #hivm.address_space<ub>> -> !tile
  %bias = tla.tensor_desc %bias_mem shape [%c1, %cn, %c1, %c1] stride [%cn, %c1, %c1, %c1] origin_shape [%c1, %cn] coord [%c0, %c0] : memref<64xf32, #hivm.address_space<ub>> -> !tile
  %out = tla.tensor_desc %out_mem shape [%c1, %cn, %c1, %c1] stride [%cn, %c1, %c1, %c1] origin_shape [%c1, %cn] coord [%c0, %c0] : memref<64xf32, #hivm.address_space<ub>> -> !tile
  "tla.vector"() ({
    "tla.vec.func"() ({
      %a = tla.load %dst : !tile -> !tla.vector<64xf32>
      %b = tla.load %factor : !tile -> !tla.vector<64xf32>
      %c = tla.load %bias : !tile -> !tla.vector<64xf32>
      %m = "tla.create_mask"() {pattern = "H", dtype = f32} : () -> !tla.mask<64>
      %result = tla.inplace_mul_add %a, %b, %c mask %m : !tla.vector<64xf32> mask !tla.mask<64>
      tla.store %out, %result : !tile, !tla.vector<64xf32>
    }) : () -> ()
  }) : () -> ()
  return
}

// CHECK: func.func private @inplace_mul_add_f32(
// CHECK-SAME: vector<256xi1>)
// CHECK-SAME: hacc.always_inline
// CHECK: func.func private @vector_region_
// CHECK: %[[DST:[a-zA-Z0-9_]+]] = {{.*}}hivm.vld
// CHECK: %[[FACTOR:[a-zA-Z0-9_]+]] = {{.*}}hivm.vld
// CHECK: %[[BIAS:[a-zA-Z0-9_]+]] = {{.*}}hivm.vld
// CHECK: call @inplace_mul_add_f32(%[[DST]], %[[FACTOR]], %[[BIAS]],
