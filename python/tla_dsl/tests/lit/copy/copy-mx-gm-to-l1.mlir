// RUN: %tla_compile %s -o - | %filecheck %s
//
// MX scale GM -> L1：A 侧两种 GM tag 都落到 zZMxScale，B 侧都落到 nNMxScale。
// 四条 bc 符号名由 C 宏展开（REGISTER_GM_TO_L1 的 tag 名 + 元素类型名）与
// TlaTensorToMemref 的路由字符串独立产生，漂移只会在上板时暴露，因此在这里锁住。
// Scale 一律是 float8_e8m0_t：REGISTER_GM_TO_L1_MX_SCALE 已并入通用宏。

module attributes {tla.module_exec_units = "cube"} {
  "tla.func"() ({
  ^bb0(%arg0: !tla.tensor<!tla.layout<!tla.shape<(128,64)>, !tla.stride<(64,1)>, !tla.shape<128,64>, RowMajor>, !tla.coord<0,0>, !tla.ptr<!tla.f8e8m0, gm, 1>>):
    %zero = "tla.make_coord"() : () -> !tla.coord<0,0>
    %og = "tla.make_shape"() : () -> !tla.shape<128,64>
    %p = "tla.tensor_ptr"(%arg0) : (!tla.tensor<!tla.layout<!tla.shape<(128,64)>, !tla.stride<(64,1)>, !tla.shape<128,64>, RowMajor>, !tla.coord<0,0>, !tla.ptr<!tla.f8e8m0, gm, 1>>) -> !tla.ptr<!tla.f8e8m0, gm, 1>
    %gs = "tla.make_shape"() : () -> !tla.shape<(1,128),(2,32)>
    %gst = "tla.make_stride"() : () -> !tla.stride<(0,64),(1,2)>
    %gl = tla.make_layout %gs, %gst origin %og {layoutTag = #tla.layout_tag<RowMajorMxScaleA>} :
      !tla.shape<(1,128),(2,32)>, !tla.stride<(0,64),(1,2)> origin !tla.shape<128,64> ->
      !tla.layout<!tla.shape<(1,128),(2,32)>, !tla.stride<(0,64),(1,2)>, !tla.shape<128,64>, RowMajorMxScaleA>
    %gm = "tla.make_tensor"(%p, %gl, %zero) :
      (!tla.ptr<!tla.f8e8m0, gm, 1>,
       !tla.layout<!tla.shape<(1,128),(2,32)>, !tla.stride<(0,64),(1,2)>, !tla.shape<128,64>, RowMajorMxScaleA>,
       !tla.coord<0,0>) -> !tla.tensor<!tla.layout<!tla.shape<(1,128),(2,32)>, !tla.stride<(0,64),(1,2)>, !tla.shape<128,64>, RowMajorMxScaleA>, !tla.coord<0,0>, !tla.ptr<!tla.f8e8m0, gm, 1>>
    %lp = "tla.alloc_ptr"() {size_bytes = 8192 : i64} : () -> !tla.ptr<!tla.f8e8m0, l1, 512>
    %ls = "tla.make_shape"() : () -> !tla.shape<(16,8),(2,32)>
    %lst = "tla.make_stride"() : () -> !tla.stride<(2,1024),(1,32)>
    %ll = tla.make_layout %ls, %lst origin %og {layoutTag = #tla.layout_tag<zZMxScale>} :
      !tla.shape<(16,8),(2,32)>, !tla.stride<(2,1024),(1,32)> origin !tla.shape<128,64> ->
      !tla.layout<!tla.shape<(16,8),(2,32)>, !tla.stride<(2,1024),(1,32)>, !tla.shape<128,64>, zZMxScale>
    %l1 = "tla.make_tensor"(%lp, %ll, %zero) :
      (!tla.ptr<!tla.f8e8m0, l1, 512>,
       !tla.layout<!tla.shape<(16,8),(2,32)>, !tla.stride<(2,1024),(1,32)>, !tla.shape<128,64>, zZMxScale>,
       !tla.coord<0,0>) -> !tla.tensor<!tla.layout<!tla.shape<(16,8),(2,32)>, !tla.stride<(2,1024),(1,32)>, !tla.shape<128,64>, zZMxScale>, !tla.coord<0,0>, !tla.ptr<!tla.f8e8m0, l1, 512>>
    "tla.cube"() ({
      "tla.copy"(%l1, %gm) : (!tla.tensor<!tla.layout<!tla.shape<(16,8),(2,32)>, !tla.stride<(2,1024),(1,32)>, !tla.shape<128,64>, zZMxScale>, !tla.coord<0,0>, !tla.ptr<!tla.f8e8m0, l1, 512>>, !tla.tensor<!tla.layout<!tla.shape<(1,128),(2,32)>, !tla.stride<(0,64),(1,2)>, !tla.shape<128,64>, RowMajorMxScaleA>, !tla.coord<0,0>, !tla.ptr<!tla.f8e8m0, gm, 1>>) -> ()
    }) : () -> ()
    "tla.return"() : () -> ()
  }) {tla.exec_units = "cube", function_type = (!tla.tensor<!tla.layout<!tla.shape<(128,64)>, !tla.stride<(64,1)>, !tla.shape<128,64>, RowMajor>, !tla.coord<0,0>, !tla.ptr<!tla.f8e8m0, gm, 1>>) -> (), sym_name = "copy_mx_gm_to_l1_A_rm"} : () -> ()
  "tla.func"() ({
  ^bb0(%arg0: !tla.tensor<!tla.layout<!tla.shape<(128,64)>, !tla.stride<(64,1)>, !tla.shape<128,64>, RowMajor>, !tla.coord<0,0>, !tla.ptr<!tla.f8e8m0, gm, 1>>):
    %zero = "tla.make_coord"() : () -> !tla.coord<0,0>
    %og = "tla.make_shape"() : () -> !tla.shape<128,64>
    %p = "tla.tensor_ptr"(%arg0) : (!tla.tensor<!tla.layout<!tla.shape<(128,64)>, !tla.stride<(64,1)>, !tla.shape<128,64>, RowMajor>, !tla.coord<0,0>, !tla.ptr<!tla.f8e8m0, gm, 1>>) -> !tla.ptr<!tla.f8e8m0, gm, 1>
    %gs = "tla.make_shape"() : () -> !tla.shape<(1,128),(2,32)>
    %gst = "tla.make_stride"() : () -> !tla.stride<(0,2),(1,256)>
    %gl = tla.make_layout %gs, %gst origin %og {layoutTag = #tla.layout_tag<ColMajorMxScaleA>} :
      !tla.shape<(1,128),(2,32)>, !tla.stride<(0,2),(1,256)> origin !tla.shape<128,64> ->
      !tla.layout<!tla.shape<(1,128),(2,32)>, !tla.stride<(0,2),(1,256)>, !tla.shape<128,64>, ColMajorMxScaleA>
    %gm = "tla.make_tensor"(%p, %gl, %zero) :
      (!tla.ptr<!tla.f8e8m0, gm, 1>,
       !tla.layout<!tla.shape<(1,128),(2,32)>, !tla.stride<(0,2),(1,256)>, !tla.shape<128,64>, ColMajorMxScaleA>,
       !tla.coord<0,0>) -> !tla.tensor<!tla.layout<!tla.shape<(1,128),(2,32)>, !tla.stride<(0,2),(1,256)>, !tla.shape<128,64>, ColMajorMxScaleA>, !tla.coord<0,0>, !tla.ptr<!tla.f8e8m0, gm, 1>>
    %lp = "tla.alloc_ptr"() {size_bytes = 8192 : i64} : () -> !tla.ptr<!tla.f8e8m0, l1, 512>
    %ls = "tla.make_shape"() : () -> !tla.shape<(16,8),(2,32)>
    %lst = "tla.make_stride"() : () -> !tla.stride<(2,1024),(1,32)>
    %ll = tla.make_layout %ls, %lst origin %og {layoutTag = #tla.layout_tag<zZMxScale>} :
      !tla.shape<(16,8),(2,32)>, !tla.stride<(2,1024),(1,32)> origin !tla.shape<128,64> ->
      !tla.layout<!tla.shape<(16,8),(2,32)>, !tla.stride<(2,1024),(1,32)>, !tla.shape<128,64>, zZMxScale>
    %l1 = "tla.make_tensor"(%lp, %ll, %zero) :
      (!tla.ptr<!tla.f8e8m0, l1, 512>,
       !tla.layout<!tla.shape<(16,8),(2,32)>, !tla.stride<(2,1024),(1,32)>, !tla.shape<128,64>, zZMxScale>,
       !tla.coord<0,0>) -> !tla.tensor<!tla.layout<!tla.shape<(16,8),(2,32)>, !tla.stride<(2,1024),(1,32)>, !tla.shape<128,64>, zZMxScale>, !tla.coord<0,0>, !tla.ptr<!tla.f8e8m0, l1, 512>>
    "tla.cube"() ({
      "tla.copy"(%l1, %gm) : (!tla.tensor<!tla.layout<!tla.shape<(16,8),(2,32)>, !tla.stride<(2,1024),(1,32)>, !tla.shape<128,64>, zZMxScale>, !tla.coord<0,0>, !tla.ptr<!tla.f8e8m0, l1, 512>>, !tla.tensor<!tla.layout<!tla.shape<(1,128),(2,32)>, !tla.stride<(0,2),(1,256)>, !tla.shape<128,64>, ColMajorMxScaleA>, !tla.coord<0,0>, !tla.ptr<!tla.f8e8m0, gm, 1>>) -> ()
    }) : () -> ()
    "tla.return"() : () -> ()
  }) {tla.exec_units = "cube", function_type = (!tla.tensor<!tla.layout<!tla.shape<(128,64)>, !tla.stride<(64,1)>, !tla.shape<128,64>, RowMajor>, !tla.coord<0,0>, !tla.ptr<!tla.f8e8m0, gm, 1>>) -> (), sym_name = "copy_mx_gm_to_l1_A_cm"} : () -> ()
  "tla.func"() ({
  ^bb0(%arg0: !tla.tensor<!tla.layout<!tla.shape<(128,64)>, !tla.stride<(64,1)>, !tla.shape<128,64>, RowMajor>, !tla.coord<0,0>, !tla.ptr<!tla.f8e8m0, gm, 1>>):
    %zero = "tla.make_coord"() : () -> !tla.coord<0,0>
    %og = "tla.make_shape"() : () -> !tla.shape<128,64>
    %p = "tla.tensor_ptr"(%arg0) : (!tla.tensor<!tla.layout<!tla.shape<(128,64)>, !tla.stride<(64,1)>, !tla.shape<128,64>, RowMajor>, !tla.coord<0,0>, !tla.ptr<!tla.f8e8m0, gm, 1>>) -> !tla.ptr<!tla.f8e8m0, gm, 1>
    %gs = "tla.make_shape"() : () -> !tla.shape<(2,64),(1,64)>
    %gst = "tla.make_stride"() : () -> !tla.stride<(1,128),(0,2)>
    %gl = tla.make_layout %gs, %gst origin %og {layoutTag = #tla.layout_tag<RowMajorMxScaleB>} :
      !tla.shape<(2,64),(1,64)>, !tla.stride<(1,128),(0,2)> origin !tla.shape<128,64> ->
      !tla.layout<!tla.shape<(2,64),(1,64)>, !tla.stride<(1,128),(0,2)>, !tla.shape<128,64>, RowMajorMxScaleB>
    %gm = "tla.make_tensor"(%p, %gl, %zero) :
      (!tla.ptr<!tla.f8e8m0, gm, 1>,
       !tla.layout<!tla.shape<(2,64),(1,64)>, !tla.stride<(1,128),(0,2)>, !tla.shape<128,64>, RowMajorMxScaleB>,
       !tla.coord<0,0>) -> !tla.tensor<!tla.layout<!tla.shape<(2,64),(1,64)>, !tla.stride<(1,128),(0,2)>, !tla.shape<128,64>, RowMajorMxScaleB>, !tla.coord<0,0>, !tla.ptr<!tla.f8e8m0, gm, 1>>
    %lp = "tla.alloc_ptr"() {size_bytes = 8192 : i64} : () -> !tla.ptr<!tla.f8e8m0, l1, 512>
    %ls = "tla.make_shape"() : () -> !tla.shape<(2,64),(16,8)>
    %lst = "tla.make_stride"() : () -> !tla.stride<(1,32),(2,1024)>
    %ll = tla.make_layout %ls, %lst origin %og {layoutTag = #tla.layout_tag<nNMxScale>} :
      !tla.shape<(2,64),(16,8)>, !tla.stride<(1,32),(2,1024)> origin !tla.shape<128,64> ->
      !tla.layout<!tla.shape<(2,64),(16,8)>, !tla.stride<(1,32),(2,1024)>, !tla.shape<128,64>, nNMxScale>
    %l1 = "tla.make_tensor"(%lp, %ll, %zero) :
      (!tla.ptr<!tla.f8e8m0, l1, 512>,
       !tla.layout<!tla.shape<(2,64),(16,8)>, !tla.stride<(1,32),(2,1024)>, !tla.shape<128,64>, nNMxScale>,
       !tla.coord<0,0>) -> !tla.tensor<!tla.layout<!tla.shape<(2,64),(16,8)>, !tla.stride<(1,32),(2,1024)>, !tla.shape<128,64>, nNMxScale>, !tla.coord<0,0>, !tla.ptr<!tla.f8e8m0, l1, 512>>
    "tla.cube"() ({
      "tla.copy"(%l1, %gm) : (!tla.tensor<!tla.layout<!tla.shape<(2,64),(16,8)>, !tla.stride<(1,32),(2,1024)>, !tla.shape<128,64>, nNMxScale>, !tla.coord<0,0>, !tla.ptr<!tla.f8e8m0, l1, 512>>, !tla.tensor<!tla.layout<!tla.shape<(2,64),(1,64)>, !tla.stride<(1,128),(0,2)>, !tla.shape<128,64>, RowMajorMxScaleB>, !tla.coord<0,0>, !tla.ptr<!tla.f8e8m0, gm, 1>>) -> ()
    }) : () -> ()
    "tla.return"() : () -> ()
  }) {tla.exec_units = "cube", function_type = (!tla.tensor<!tla.layout<!tla.shape<(128,64)>, !tla.stride<(64,1)>, !tla.shape<128,64>, RowMajor>, !tla.coord<0,0>, !tla.ptr<!tla.f8e8m0, gm, 1>>) -> (), sym_name = "copy_mx_gm_to_l1_B_rm"} : () -> ()
  "tla.func"() ({
  ^bb0(%arg0: !tla.tensor<!tla.layout<!tla.shape<(128,64)>, !tla.stride<(64,1)>, !tla.shape<128,64>, RowMajor>, !tla.coord<0,0>, !tla.ptr<!tla.f8e8m0, gm, 1>>):
    %zero = "tla.make_coord"() : () -> !tla.coord<0,0>
    %og = "tla.make_shape"() : () -> !tla.shape<128,64>
    %p = "tla.tensor_ptr"(%arg0) : (!tla.tensor<!tla.layout<!tla.shape<(128,64)>, !tla.stride<(64,1)>, !tla.shape<128,64>, RowMajor>, !tla.coord<0,0>, !tla.ptr<!tla.f8e8m0, gm, 1>>) -> !tla.ptr<!tla.f8e8m0, gm, 1>
    %gs = "tla.make_shape"() : () -> !tla.shape<(2,64),(1,64)>
    %gst = "tla.make_stride"() : () -> !tla.stride<(1,2),(0,64)>
    %gl = tla.make_layout %gs, %gst origin %og {layoutTag = #tla.layout_tag<ColMajorMxScaleB>} :
      !tla.shape<(2,64),(1,64)>, !tla.stride<(1,2),(0,64)> origin !tla.shape<128,64> ->
      !tla.layout<!tla.shape<(2,64),(1,64)>, !tla.stride<(1,2),(0,64)>, !tla.shape<128,64>, ColMajorMxScaleB>
    %gm = "tla.make_tensor"(%p, %gl, %zero) :
      (!tla.ptr<!tla.f8e8m0, gm, 1>,
       !tla.layout<!tla.shape<(2,64),(1,64)>, !tla.stride<(1,2),(0,64)>, !tla.shape<128,64>, ColMajorMxScaleB>,
       !tla.coord<0,0>) -> !tla.tensor<!tla.layout<!tla.shape<(2,64),(1,64)>, !tla.stride<(1,2),(0,64)>, !tla.shape<128,64>, ColMajorMxScaleB>, !tla.coord<0,0>, !tla.ptr<!tla.f8e8m0, gm, 1>>
    %lp = "tla.alloc_ptr"() {size_bytes = 8192 : i64} : () -> !tla.ptr<!tla.f8e8m0, l1, 512>
    %ls = "tla.make_shape"() : () -> !tla.shape<(2,64),(16,8)>
    %lst = "tla.make_stride"() : () -> !tla.stride<(1,32),(2,1024)>
    %ll = tla.make_layout %ls, %lst origin %og {layoutTag = #tla.layout_tag<nNMxScale>} :
      !tla.shape<(2,64),(16,8)>, !tla.stride<(1,32),(2,1024)> origin !tla.shape<128,64> ->
      !tla.layout<!tla.shape<(2,64),(16,8)>, !tla.stride<(1,32),(2,1024)>, !tla.shape<128,64>, nNMxScale>
    %l1 = "tla.make_tensor"(%lp, %ll, %zero) :
      (!tla.ptr<!tla.f8e8m0, l1, 512>,
       !tla.layout<!tla.shape<(2,64),(16,8)>, !tla.stride<(1,32),(2,1024)>, !tla.shape<128,64>, nNMxScale>,
       !tla.coord<0,0>) -> !tla.tensor<!tla.layout<!tla.shape<(2,64),(16,8)>, !tla.stride<(1,32),(2,1024)>, !tla.shape<128,64>, nNMxScale>, !tla.coord<0,0>, !tla.ptr<!tla.f8e8m0, l1, 512>>
    "tla.cube"() ({
      "tla.copy"(%l1, %gm) : (!tla.tensor<!tla.layout<!tla.shape<(2,64),(16,8)>, !tla.stride<(1,32),(2,1024)>, !tla.shape<128,64>, nNMxScale>, !tla.coord<0,0>, !tla.ptr<!tla.f8e8m0, l1, 512>>, !tla.tensor<!tla.layout<!tla.shape<(2,64),(1,64)>, !tla.stride<(1,2),(0,64)>, !tla.shape<128,64>, ColMajorMxScaleB>, !tla.coord<0,0>, !tla.ptr<!tla.f8e8m0, gm, 1>>) -> ()
    }) : () -> ()
    "tla.return"() : () -> ()
  }) {tla.exec_units = "cube", function_type = (!tla.tensor<!tla.layout<!tla.shape<(128,64)>, !tla.stride<(64,1)>, !tla.shape<128,64>, RowMajor>, !tla.coord<0,0>, !tla.ptr<!tla.f8e8m0, gm, 1>>) -> (), sym_name = "copy_mx_gm_to_l1_B_cm"} : () -> ()
}

// CHECK-DAG: func.func private @copy_gm_ColMajorMxScaleB_to_l1_nNMxScale_float8_e8m0_t
// CHECK-DAG: func.func private @copy_gm_RowMajorMxScaleB_to_l1_nNMxScale_float8_e8m0_t
// CHECK-DAG: func.func private @copy_gm_ColMajorMxScaleA_to_l1_zZMxScale_float8_e8m0_t
// CHECK-DAG: func.func private @copy_gm_RowMajorMxScaleA_to_l1_zZMxScale_float8_e8m0_t
// CHECK-NOT: "tla.copy"
