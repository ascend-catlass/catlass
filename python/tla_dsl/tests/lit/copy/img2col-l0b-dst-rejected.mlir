// L0B stores the img2col window in nZ. Reject zN even when
// the IR bypasses the Python frontend.

// RUN: not %tla_compile %s -o - 2>&1 | %filecheck %s --check-prefix=L0BDST
// L0BDST: img2col destination must be an l0a tile tagged zN or an l0b tile tagged nZ

module {
  tla.func @img2col_min(%arg0: !tla.tensor<!tla.layout<!tla.shape<64,288>, !tla.stride<288,1>, !tla.shape<64,288>, RowMajor>, !tla.coord<0,0>, !tla.ptr<f16, gm, 2>>) {
    %0 = tla.alloc_ptr{size_bytes = 4096} -> !tla.ptr<f16, l1, 512>
    %1 = tla.alloc_ptr{size_bytes = 36864} -> !tla.ptr<f16, l0b, 512>
    tla.cube {
      %2 = tla.tensor_ptr %arg0 : !tla.tensor<!tla.layout<!tla.shape<64,288>, !tla.stride<288,1>, !tla.shape<64,288>, RowMajor>, !tla.coord<0,0>, !tla.ptr<f16, gm, 2>> -> !tla.ptr<f16, gm, 2>
      %3 = tla.make_shape -> !tla.shape<(64,1),(16,2)>
      %4 = tla.make_stride -> !tla.stride<(16,1024),(1,1024)>
      %5 = tla.make_shape -> !tla.shape<64,32>
      %6 = tla.make_layout %3, %4 origin %5 {layoutTag = #tla.layout_tag<zNUnAlign>} : !tla.shape<(64,1),(16,2)>, !tla.stride<(16,1024),(1,1024)> origin !tla.shape<64,32> -> !tla.layout<!tla.shape<(64,1),(16,2)>, !tla.stride<(16,1024),(1,1024)>, !tla.shape<64,32>, zNUnAlign>
      %7 = tla.make_coord -> !tla.coord<0,0>
      %8 = tla.make_tensor %2, %6, %7 : !tla.ptr<f16, gm, 2>, !tla.layout<!tla.shape<(64,1),(16,2)>, !tla.stride<(16,1024),(1,1024)>, !tla.shape<64,32>, zNUnAlign>, !tla.coord<0,0> -> !tla.tensor<!tla.layout<!tla.shape<(64,1),(16,2)>, !tla.stride<(16,1024),(1,1024)>, !tla.shape<64,32>, zNUnAlign>, !tla.coord<0,0>, !tla.ptr<f16, gm, 2>>
      %9 = tla.make_tensor_like %0 like %8 layoutTag(<zNUnAlign>) : !tla.ptr<f16, l1, 512>, !tla.tensor<!tla.layout<!tla.shape<(64,1),(16,2)>, !tla.stride<(16,1024),(1,1024)>, !tla.shape<64,32>, zNUnAlign>, !tla.coord<0,0>, !tla.ptr<f16, gm, 2>> -> !tla.tensor<!tla.layout<!tla.shape<(64,1),(16,2)>, !tla.stride<(16,1024),(1,1024)>, !tla.shape<64,32>, zNUnAlign>, !tla.coord<0,0>, !tla.ptr<f16, l1, 512>>
      tla.copy %9, %8 : !tla.tensor<!tla.layout<!tla.shape<(64,1),(16,2)>, !tla.stride<(16,1024),(1,1024)>, !tla.shape<64,32>, zNUnAlign>, !tla.coord<0,0>, !tla.ptr<f16, l1, 512>>, !tla.tensor<!tla.layout<!tla.shape<(64,1),(16,2)>, !tla.stride<(16,1024),(1,1024)>, !tla.shape<64,32>, zNUnAlign>, !tla.coord<0,0>, !tla.ptr<f16, gm, 2>>
      %10 = tla.tensor_ptr %arg0 : !tla.tensor<!tla.layout<!tla.shape<64,288>, !tla.stride<288,1>, !tla.shape<64,288>, RowMajor>, !tla.coord<0,0>, !tla.ptr<f16, gm, 2>> -> !tla.ptr<f16, gm, 2>
      %11 = tla.make_shape -> !tla.shape<64,288>
      %12 = tla.make_stride -> !tla.stride<288,1>
      %13 = tla.make_layout %11, %12 : !tla.shape<64,288>, !tla.stride<288,1> -> !tla.layout<!tla.shape<64,288>, !tla.stride<288,1>, !tla.shape<64,288>, RowMajor>
      %14 = tla.make_coord -> !tla.coord<0,0>
      %15 = tla.make_tensor %10, %13, %14 : !tla.ptr<f16, gm, 2>, !tla.layout<!tla.shape<64,288>, !tla.stride<288,1>, !tla.shape<64,288>, RowMajor>, !tla.coord<0,0> -> !tla.tensor<!tla.layout<!tla.shape<64,288>, !tla.stride<288,1>, !tla.shape<64,288>, RowMajor>, !tla.coord<0,0>, !tla.ptr<f16, gm, 2>>
      %16 = tla.make_tensor_like %1 like %15 layoutTag(<zN>) : !tla.ptr<f16, l0b, 512>, !tla.tensor<!tla.layout<!tla.shape<64,288>, !tla.stride<288,1>, !tla.shape<64,288>, RowMajor>, !tla.coord<0,0>, !tla.ptr<f16, gm, 2>> -> !tla.tensor<!tla.layout<!tla.shape<(16,4),(16,18)>, !tla.stride<(16,256),(1,1024)>, !tla.shape<64,288>, zN>, !tla.coord<0,0>, !tla.ptr<f16, l0b, 512>>
      %c8 = arith.constant 8 : index
      %c8_0 = arith.constant 8 : index
      %c3 = arith.constant 3 : index
      %c3_1 = arith.constant 3 : index
      %c1 = arith.constant 1 : index
      %c1_2 = arith.constant 1 : index
      %c1_3 = arith.constant 1 : index
      %c1_4 = arith.constant 1 : index
      %c1_5 = arith.constant 1 : index
      %c1_6 = arith.constant 1 : index
      %c1_7 = arith.constant 1 : index
      %c1_8 = arith.constant 1 : index
      tla.img2col %16, %9, fmap(%c8, %c8_0) filter(%c3, %c3_1) stride(%c1, %c1_2) dilation(%c1_3, %c1_4) pad(%c1_5, %c1_6, %c1_7, %c1_8) {pad_value = 0 : i32} : !tla.tensor<!tla.layout<!tla.shape<(16,4),(16,18)>, !tla.stride<(16,256),(1,1024)>, !tla.shape<64,288>, zN>, !tla.coord<0,0>, !tla.ptr<f16, l0b, 512>>, !tla.tensor<!tla.layout<!tla.shape<(64,1),(16,2)>, !tla.stride<(16,1024),(1,1024)>, !tla.shape<64,32>, zNUnAlign>, !tla.coord<0,0>, !tla.ptr<f16, l1, 512>>
    }
    tla.return
  }
}
