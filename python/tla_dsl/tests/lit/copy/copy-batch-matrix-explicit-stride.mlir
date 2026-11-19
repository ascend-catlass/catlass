// RUN: %tla_compile %s -o - | %filecheck %s
// A batch with a singleton matrix row must retain its explicit dynamic
// leading matrix stride and independent source batch stride.
// CHECK-LABEL: func.func @repro
// CHECK-DAG: %[[SRCSTEP:.*]] = llvm.mlir.constant(128 : i64)
// CHECK-DAG: %[[COUNT:.*]] = llvm.mlir.constant(2 : i64)
// CHECK-DAG: %[[DISABLE:.*]] = llvm.mlir.constant(0 : i64)
// CHECK: call @copy_gm_RowMajor_to_l1_zN_fp8_e4m3fn_t_batch({{.*}}, %[[COUNT]], %[[SRCSTEP]], {{.*}}, %[[DISABLE]]) :
// CHECK-NOT: tla.tensor_desc

module {
  tla.func @repro(%extent: index, %step: index) {
    %address = arith.constant 4096 : i64
    %ptr = tla.inttoptr %address : i64 -> !tla.ptr<f8E4M3FN, gm, 1>
    %s = tla.make_shape %extent -> !tla.shape<2,1,?>
    %st = tla.make_stride %step -> !tla.stride<128,?,1>
    %l = tla.make_layout %s, %st : !tla.shape<2,1,?>, !tla.stride<128,?,1> -> !tla.layout<!tla.shape<2,1,?>, !tla.stride<128,?,1>, !tla.shape<2,1,?>, RowMajor>
    %c = tla.make_coord -> !tla.coord<0,0,0>
    %tensor = tla.make_tensor %ptr, %l, %c : !tla.ptr<f8E4M3FN, gm, 1>, !tla.layout<!tla.shape<2,1,?>, !tla.stride<128,?,1>, !tla.shape<2,1,?>, RowMajor>, !tla.coord<0,0,0> -> !tla.tensor<!tla.layout<!tla.shape<2,1,?>, !tla.stride<128,?,1>, !tla.shape<2,1,?>, RowMajor>, !tla.coord<0,0,0>, !tla.ptr<f8E4M3FN, gm, 1>>
    %dp = tla.alloc_ptr {size_bytes = 2048} -> !tla.ptr<f8E4M3FN, l1, 512>
    %ds = tla.make_shape -> !tla.shape<2,(16,1),(32,2)>
    %dst = tla.make_stride -> !tla.stride<1024,(32,512),(1,512)>
    %do = tla.make_shape %extent -> !tla.shape<2,1,?>
    %dl = tla.make_layout %ds, %dst origin %do {layoutTag = #tla.layout_tag<zN>} : !tla.shape<2,(16,1),(32,2)>, !tla.stride<1024,(32,512),(1,512)> origin !tla.shape<2,1,?> -> !tla.layout<!tla.shape<2,(16,1),(32,2)>, !tla.stride<1024,(32,512),(1,512)>, !tla.shape<2,1,?>, zN>
    %dc = tla.make_coord -> !tla.coord<0,0,0>
    %d = tla.make_tensor %dp, %dl, %dc : !tla.ptr<f8E4M3FN, l1, 512>, !tla.layout<!tla.shape<2,(16,1),(32,2)>, !tla.stride<1024,(32,512),(1,512)>, !tla.shape<2,1,?>, zN>, !tla.coord<0,0,0> -> !tla.tensor<!tla.layout<!tla.shape<2,(16,1),(32,2)>, !tla.stride<1024,(32,512),(1,512)>, !tla.shape<2,1,?>, zN>, !tla.coord<0,0,0>, !tla.ptr<f8E4M3FN, l1, 512>>
    tla.cube {
      tla.copy %d, %tensor {src_cache_policy = #tla.cache_policy<disable>} : !tla.tensor<!tla.layout<!tla.shape<2,(16,1),(32,2)>, !tla.stride<1024,(32,512),(1,512)>, !tla.shape<2,1,?>, zN>, !tla.coord<0,0,0>, !tla.ptr<f8E4M3FN, l1, 512>>, !tla.tensor<!tla.layout<!tla.shape<2,1,?>, !tla.stride<128,?,1>, !tla.shape<2,1,?>, RowMajor>, !tla.coord<0,0,0>, !tla.ptr<f8E4M3FN, gm, 1>>
    }
    tla.return
  }
}
