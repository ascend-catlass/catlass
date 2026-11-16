// RUN: %tla_compile %s --mlir-print-ir-after=tla-vector-region-simd -o %t > %t.ir 2>&1
// RUN: %filecheck %s --implicit-check-not=tla.scatter < %t.ir

!layout = !tla.layout<!tla.shape<64>, !tla.stride<1>, !tla.shape<64>, RowMajor>
!coord = !tla.coord<0>
!data = !tla.tensor<!layout, !coord, !tla.ptr<i64, ub, 32>>
!udata = !tla.tensor<!layout, !coord, !tla.ptr<ui64, ub, 32>>
!index = !tla.tensor<!layout, !coord, !tla.ptr<i32, ub, 32>>

module {
  func.func @scatter_b64(%mem: memref<64xi64, #hivm.address_space<ub>>, %umem: memref<64xui64, #hivm.address_space<ub>>, %imem: memref<64xi32, #hivm.address_space<ub>>) {
    %zero = arith.constant 0 : index
    %one = arith.constant 1 : index
    %size = arith.constant 64 : index
    %dst = tla.tensor_desc %mem shape [%one, %size, %one, %one] stride [%size, %one, %one, %one] origin_shape [%one, %size] coord [%zero, %zero] : memref<64xi64, #hivm.address_space<ub>> -> !data
    %udst = tla.tensor_desc %umem shape [%one, %size, %one, %one] stride [%size, %one, %one, %one] origin_shape [%one, %size] coord [%zero, %zero] : memref<64xui64, #hivm.address_space<ub>> -> !udata
    %idx = tla.tensor_desc %imem shape [%one, %size, %one, %one] stride [%size, %one, %one, %one] origin_shape [%one, %size] coord [%zero, %zero] : memref<64xi32, #hivm.address_space<ub>> -> !index
    "tla.vector"() ({
      "tla.vec.func"() ({
        %short = tla.load %dst : !data -> !tla.vector<17xi64>
        %src = tla.load %dst : !data -> !tla.vector<32xi64>
        %signed = tla.load %idx : !index -> !tla.vector<64xi32>
        %i32 = tla.bitcast %signed : !tla.vector<64xi32> -> !tla.vector<64xui32>
        %i64 = tla.bitcast %src : !tla.vector<32xi64> -> !tla.vector<32xui64>
        %vl2 = "tla.create_mask"() {pattern = "VL2", dtype = i64} : () -> !tla.mask<32>
        %half = "tla.create_mask"() {pattern = "H", dtype = i64} : () -> !tla.mask<32>
        %three = arith.constant 3 : index
        %tail, %remaining = tla.update_mask %three, i64 : !tla.mask<32>, index
        %raw = tla.load %idx : !index -> !tla.mask<32>
        %mixed = "tla.bitwise_and"(%raw, %tail) : (!tla.mask<32>, !tla.mask<32>) -> !tla.mask<32>
        %neg = "tla.bitwise_not"(%mixed) : (!tla.mask<32>) -> !tla.mask<32>
        %selected = "tla.where"(%vl2, %mixed, %neg) : (!tla.mask<32>, !tla.mask<32>, !tla.mask<32>) -> !tla.mask<32>
        tla.store %dst, %src mask %selected : !data, !tla.vector<32xi64> mask !tla.mask<32>
        tla.scatter %dst, %short, %i32 : !data, !tla.vector<17xi64>, !tla.vector<64xui32>
        tla.scatter %dst, %src, %i32 mask %vl2 : !data, !tla.vector<32xi64>, !tla.vector<64xui32> mask !tla.mask<32>
        tla.scatter %dst, %src, %i64 mask %half : !data, !tla.vector<32xi64>, !tla.vector<32xui64> mask !tla.mask<32>
        tla.scatter %udst, %i64, %i32 mask %selected : !udata, !tla.vector<32xui64>, !tla.vector<64xui32> mask !tla.mask<32>
        tla.scatter %udst, %i64, %i64 : !udata, !tla.vector<32xui64>, !tla.vector<32xui64>
      }) {mode = "simd"} : () -> ()
    }) : () -> ()
    return
  }
}

// CHECK-DAG: func.func private @scatter_int64_t_u32(memref<?xi64, strided<[?], offset: ?>, #hivm.address_space<ub>>, vector<32xi64>, vector<64xui32>, vector<256xi1>) attributes {{.*}}llvm.emit_c_interface
// CHECK-DAG: func.func private @scatter_int64_t_u64(memref<?xi64, {{.*}}, vector<32xi64>, vector<32xui64>, vector<256xi1>)
// CHECK-DAG: func.func private @scatter_uint64_t_u32(memref<?xui64, {{.*}}, vector<32xui64>, vector<64xui32>, vector<256xi1>)
// CHECK-DAG: func.func private @scatter_uint64_t_u64(memref<?xui64, {{.*}}, vector<32xui64>, vector<32xui64>, vector<256xi1>)
// CHECK-LABEL: func.func private @vector_region_0
// CHECK: %[[VL:.*]] = ave.hir.pge <VL2>
// CHECK: %[[VM:.*]] = "hivm_regbaseintrins.intr.hivm.punpack"(%[[VL]]
// CHECK: %[[H:.*]] = ave.hir.pge <VL16>
// CHECK: %[[HM:.*]] = "hivm_regbaseintrins.intr.hivm.punpack"(%[[H]]
// CHECK: %[[P:.*]], {{.*}} = ave.hir.plt
// CHECK: %[[TM:.*]] = "hivm_regbaseintrins.intr.hivm.punpack"(%[[P]]
// CHECK: %[[SELECT:.*]] = call @tla_mask_select
// CHECK: ave.hir.masked_store{{.*}}%[[SELECT]]
// CHECK: call @scatter_int64_t_u32
// CHECK: call @scatter_int64_t_u32({{.*}}, %[[VM]])
// CHECK: call @scatter_int64_t_u64({{.*}}, %[[HM]])
// CHECK: call @scatter_uint64_t_u32({{.*}}, %[[SELECT]])
// CHECK: call @scatter_uint64_t_u64
