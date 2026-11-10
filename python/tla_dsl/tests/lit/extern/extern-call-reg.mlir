// RUN: %tla_compile %s --mlir-print-ir-after=tla-lower-ptr -o %t 2>&1 | %filecheck %s --check-prefix=PTR
// RUN: %tla_compile %s --mlir-print-ir-after=tla-vector-region-simd -o %t 2>&1 | %filecheck %s --check-prefix=SIMD

!tile = !tla.tensor<!tla.layout<!tla.shape<64>, !tla.stride<1>, !tla.shape<64>, RowMajor>, !tla.coord<0>, !tla.ptr<f32, ub, 4>>
!reg = !tla.vector<64xf32>

module {
  func.func @register_flow(%dst: memref<64xf32, #hivm.address_space<ub>>,
                          %addr: i64, %scalar: f32, %n: index, %active: i1)
      attributes {hivm.func_core_type = #hivm.func_core_type<AIV>} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c64 = arith.constant 64 : index
    %ptr = tla.inttoptr %addr : i64 -> !tla.ptr<f32, ub, 4>
    %tensor = tla.tensor_desc %dst shape [%c1, %c64, %c1, %c1] stride [%c64, %c1, %c1, %c1] origin_shape [%c1, %c64] coord [%c0, %c0] : memref<64xf32, #hivm.address_space<ub>> -> !tile
    "tla.vec.func"() ({
      %shape = "tla.make_shape"() : () -> !tla.shape<64>
      %coord = "tla.make_coord"() : () -> !tla.coord<0>
      %tile = "tla.tile_view"(%tensor, %shape, %coord) : (!tile, !tla.shape<64>, !tla.coord<0>) -> !tile
      %x = tla.load %tile : !tile -> !reg
      %seed = tla.call_extern @mixed(%x, %ptr, %scalar) : (!reg, !tla.ptr<f32, ub, 4>, f32) -> !reg
      %loop = scf.for %i = %c0 to %n step %c1 iter_args(%acc = %seed) -> !reg {
        %next = scf.if %active -> !reg {
          %sum = tla.call_extern @add_reg(%acc, %x) : (!reg, !reg) -> !reg
          scf.yield %sum : !reg
        } else {
          %sum = tla.call_extern @mixed(%acc, %ptr, %scalar) : (!reg, !tla.ptr<f32, ub, 4>, f32) -> !reg
          scf.yield %sum : !reg
        }
        scf.yield %next : !reg
      }
      tla.call_extern @consume_reg(%loop) : (!reg) -> ()
      %abs = tla.abs %loop : !reg -> !reg
      tla.store %tile, %abs : !tile, !reg
    }) : () -> ()
    return
  }
}

// PTR-DAG: func.func private @mixed(!tla.vector<64xf32>, i64, f32) -> !tla.vector<64xf32>
// PTR: %[[VALUE:[^ ]+]] = tla.call_extern @mixed({{.*}}) : (!tla.vector<64xf32>, i64, f32) -> !tla.vector<64xf32>
// PTR: scf.for {{.*}}iter_args({{.*}} = %[[VALUE]])

// SIMD-DAG: func.func private @mixed(vector<64xf32>, i64, f32) -> vector<64xf32>
// SIMD-DAG: func.func private @add_reg(vector<64xf32>, vector<64xf32>) -> vector<64xf32>
// SIMD-DAG: func.func private @consume_reg(vector<64xf32>)
// SIMD-LABEL: func.func @register_flow
// SIMD: call @[[VF:[^(]+]]({{.*}}) {{.*}}hivm.vector_function
// SIMD: func.func private @[[VF]](
// SIMD-SAME: hivm.vector_function
// SIMD: %[[SEED:[^ ]+]] = call @mixed({{.*}}) : (vector<64xf32>, i64, f32) -> vector<64xf32>
// SIMD: %[[LOOP:[^ ]+]] = scf.for {{.*}}iter_args({{.*}} = %[[SEED]])
// SIMD: scf.if {{.*}} -> (vector<64xf32>)
// SIMD: %[[ADD:[^ ]+]] = func.call @add_reg
// SIMD: scf.yield %[[ADD]] : vector<64xf32>
// SIMD: %[[MIX:[^ ]+]] = func.call @mixed
// SIMD: scf.yield %[[MIX]] : vector<64xf32>
// SIMD: call @consume_reg(%[[LOOP]]) : (vector<64xf32>) -> ()
// SIMD: ave.hir.vabs %[[LOOP]]
// SIMD: ave.hir.masked_store
