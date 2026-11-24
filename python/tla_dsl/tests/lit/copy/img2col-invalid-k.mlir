// Exercise the verifier directly, without Python frontend validation.
// A K extent that is not a whole number of C0 blocks: the C0 leaf itself is 24.
// RUN: sed 's/(16,18)/(24,12)/g' %S/img2col-l1-to-l0a-half.mlir > %t.extent.mlir
// RUN: not %tla_compile %t.extent.mlir -o - 2>&1 | %filecheck %s --check-prefix=EXTENT
// EXTENT: destination storage must have static 16-element K blocks
// RUN: sed 's/arith.constant 3 : index/arith.constant 0 : index/g' %S/img2col-l1-to-l0a-half.mlir > %t.filter.mlir
// RUN: not %tla_compile %t.filter.mlir -o - 2>&1 | %filecheck %s --check-prefix=FILTER
// FILTER: filter_hw entries must be in 1..511
// A K start that is not a multiple of C0. C0-aligned starts inside a C1 slice
// are legal (see img2col-sub-slice-k.mlir); 8 is not one.
// RUN: sed 's/!tla.coord<0,0>/!tla.coord<0,8>/g' %S/img2col-l1-to-l0a-half.mlir > %t.coord.mlir
// RUN: not %tla_compile %t.coord.mlir -o - 2>&1 | %filecheck %s --check-prefix=COORD
// COORD: destination K coordinate must be static, within 0..65535 and aligned to 16 elements
// RUN: sed 's/arith.constant 3 : index/arith.addi %c8, %c8 : index/g' %S/img2col-l1-to-l0a-half.mlir > %t.dynamic.mlir
// RUN: not %tla_compile %t.dynamic.mlir -o - 2>&1 | %filecheck %s --check-prefix=DYNAMIC
// DYNAMIC: filter_hw must be compile-time constants for K-window validation
// RUN: sed 's/(16,18)/(16,27)/g; s/64,288/64,432/g' %S/img2col-l1-to-l0a-half.mlir > %t.bounds.mlir
// RUN: not %tla_compile %t.bounds.mlir -o - 2>&1 | %filecheck %s --check-prefix=BOUNDS
// BOUNDS: destination K window exceeds the source channel/filter extent
// RUN: sed 's/%c8 = arith.constant 8 : index/%c8 = arith.constant 32768 : index/' %S/img2col-l1-to-l0a-half.mlir > %t.fmap-range.mlir
// RUN: not %tla_compile %t.fmap-range.mlir -o - 2>&1 | %filecheck %s --check-prefix=FMAP-RANGE
// FMAP-RANGE: fmap_h must be in 1..32767
// RUN: sed 's/%c1 = arith.constant 1 : index/%c1 = arith.constant 0 : index/' %S/img2col-l1-to-l0a-half.mlir > %t.stride-range.mlir
// RUN: not %tla_compile %t.stride-range.mlir -o - 2>&1 | %filecheck %s --check-prefix=STRIDE-RANGE
// STRIDE-RANGE: stride_h must be in 1..63
// RUN: sed 's/%c1_2 = arith.constant 1 : index/%c1_2 = arith.constant 64 : index/' %S/img2col-l1-to-l0a-half.mlir > %t.stride-upper.mlir
// RUN: not %tla_compile %t.stride-upper.mlir -o - 2>&1 | %filecheck %s --check-prefix=STRIDE-UPPER
// STRIDE-UPPER: stride_w must be in 1..63
// RUN: sed 's/%c8_0 = arith.constant 8 : index/%c8_0 = arith.constant 4 : index/' %S/img2col-l1-to-l0a-half.mlir > %t.src-pitch.mlir
// RUN: not %tla_compile %t.src-pitch.mlir -o - 2>&1 | %filecheck %s --check-prefix=SRC-PITCH
// SRC-PITCH: source C1 stride must equal fmap_h * fmap_w * C0 for multiple C1 slices
// A first-C1 K window still requires a packed multi-C1 source. These 32 spatial
// positions require a stride of 512 elements, while the view carries 1024.
// RUN: sed 's/(16,18)/(16,1)/g; s/64,288/64,16/g; s/(64,1),(16,2)/(32,1),(16,2)/g; s/64,32/32,32/g; s/%c8 = arith.constant 8 : index/%c8 = arith.constant 1 : index/; s/%c8_0 = arith.constant 8 : index/%c8_0 = arith.constant 32 : index/; s/arith.constant 3 : index/arith.constant 1 : index/g' %S/img2col-l1-to-l0a-half.mlir > %t.first-c1-pitch.mlir
// RUN: not %tla_compile %t.first-c1-pitch.mlir -o - 2>&1 | %filecheck %s --check-prefix=SRC-PITCH
// RUN: sed 's/%c8_0 = arith.constant 8 : index/%c8_0 = arith.constant 9 : index/' %S/img2col-l1-to-l0a-half.mlir > %t.src-extent.mlir
// RUN: not %tla_compile %t.src-extent.mlir -o - 2>&1 | %filecheck %s --check-prefix=SRC-EXTENT
// SRC-EXTENT: fmap_hw exceeds the number of spatial positions in the source view
// RUN: sed 's/!tla.coord<0,0>/!tla.coord<32768,0>/g' %S/img2col-l1-to-l0a-half.mlir > %t.m-start.mlir
// RUN: not %tla_compile %t.m-start.mlir -o - 2>&1 | %filecheck %s --check-prefix=M-START
// M-START: destination M coordinate must be within 0..32767
// RUN: sed 's/%c1_3 = arith.constant 1 : index/%c1_3 = arith.constant 256 : index/' %S/img2col-l1-to-l0a-half.mlir > %t.dilation-range.mlir
// RUN: not %tla_compile %t.dilation-range.mlir -o - 2>&1 | %filecheck %s --check-prefix=DILATION-RANGE
// DILATION-RANGE: dilation_h must be in 1..255
// RUN: sed 's/%c1_5 = arith.constant 1 : index/%c1_5 = arith.constant 256 : index/' %S/img2col-l1-to-l0a-half.mlir > %t.pad-range.mlir
// RUN: not %tla_compile %t.pad-range.mlir -o - 2>&1 | %filecheck %s --check-prefix=PAD-RANGE
// PAD-RANGE: pad_left must be in 0..255
// RUN: sed '/tla.copy %9, %8/d; s/zNUnAlign/zN/g' %S/img2col-l1-to-l0a-half.mlir > %t.src-layout.mlir
// RUN: not %tla_compile %t.src-layout.mlir -o - 2>&1 | %filecheck %s --check-prefix=SRC-LAYOUT
// SRC-LAYOUT: img2col source must be tagged zNUnAlign
// RUN: sed 's/layoutTag(<zN>)/layoutTag(<nZ>)/; s/, zN>, !tla.coord<0,0>, !tla.ptr<f16, l0a, 512>>/, nZ>, !tla.coord<0,0>, !tla.ptr<f16, l0a, 512>>/g' %S/img2col-l1-to-l0a-half.mlir > %t.dst-layout.mlir
// RUN: not %tla_compile %t.dst-layout.mlir -o - 2>&1 | %filecheck %s --check-prefix=DST-LAYOUT
// DST-LAYOUT: img2col destination must be an l0a tile tagged zN or an l0b tile tagged nZ
// RUN: sed 's/(1,1024)>, !tla.shape<64,288>, zN>/(1,1023)>, !tla.shape<64,288>, zN>/g' %S/img2col-l1-to-l0a-half.mlir > %t.dst-stride.mlir
// RUN: not %tla_compile %t.dst-stride.mlir -o - 2>&1 | %filecheck %s --check-prefix=DST-STRIDE
// DST-STRIDE: destination outer stride must fit the unsigned 16-bit 512-byte-fractal field
// RUN: sed '/tla.img2col/s/{pad_value = 0 : i32}/{pad_value = 0 : i32, repeat_mode = 1 : i32, repeat_stride = 1 : i32, repeat_time = 2 : i32}/' %S/img2col-l1-to-l0a-half.mlir > %t.repeat.mlir
// RUN: not %tla_compile %t.repeat.mlir -o - 2>&1 | %filecheck %s --check-prefix=BOUNDS
// RUN: sed 's/zN>, !tla.coord<0,0>, !tla.ptr<f16, l0a/zN>, !tla.coord<1,0>, !tla.ptr<f16, l0a/g' %S/img2col-l1-to-l0a-half.mlir > %t.dst-address.mlir
// RUN: not %tla_compile %t.dst-address.mlir -o - 2>&1 | %filecheck %s --check-prefix=DST-ADDRESS
// DST-ADDRESS: effective L0 destination address must be 512-byte aligned
// RUN: sed 's/zNUnAlign>, !tla.coord<0,0>, !tla.ptr<f16, l1/zNUnAlign>, !tla.coord<0,1>, !tla.ptr<f16, l1/g' %S/img2col-l1-to-l0a-half.mlir > %t.src-address.mlir
// RUN: not %tla_compile %t.src-address.mlir -o - 2>&1 | %filecheck %s --check-prefix=SRC-ADDRESS
// SRC-ADDRESS: effective L1 source address must be 32-byte aligned
