// Exercise the verifier directly, without Python frontend validation.
// A K extent that is not a whole number of C0 blocks: the C0 leaf itself is 24.
// RUN: sed 's/(16,18)/(24,12)/g' %S/img2col-l1-to-l0a-half.mlir > %t.extent.mlir
// RUN: not %tla_compile %t.extent.mlir -o - 2>&1 | %filecheck %s --check-prefix=EXTENT
// EXTENT: destination K extent must be static, C0-aligned and within 1..65535
// RUN: sed 's/arith.constant 3 : index/arith.constant 0 : index/g' %S/img2col-l1-to-l0a-half.mlir > %t.filter.mlir
// RUN: not %tla_compile %t.filter.mlir -o - 2>&1 | %filecheck %s --check-prefix=FILTER
// FILTER: filter_hw entries must be in 1..511
// A K start that is not a multiple of C0. C0-aligned starts inside a C1 slice
// are legal (see img2col-sub-slice-k.mlir); 8 is not one.
// RUN: sed 's/!tla.coord<0,0>/!tla.coord<0,8>/g' %S/img2col-l1-to-l0a-half.mlir > %t.coord.mlir
// RUN: not %tla_compile %t.coord.mlir -o - 2>&1 | %filecheck %s --check-prefix=COORD
// COORD: destination K coordinate must be static, within 0..65535 and C0-aligned
// RUN: sed 's/arith.constant 3 : index/arith.addi %c8, %c8 : index/g' %S/img2col-l1-to-l0a-half.mlir > %t.dynamic.mlir
// RUN: not %tla_compile %t.dynamic.mlir -o - 2>&1 | %filecheck %s --check-prefix=DYNAMIC
// DYNAMIC: filter_hw must be compile-time constants for K-window validation
// RUN: sed 's/(16,18)/(16,27)/g' %S/img2col-l1-to-l0a-half.mlir > %t.bounds.mlir
// RUN: not %tla_compile %t.bounds.mlir -o - 2>&1 | %filecheck %s --check-prefix=BOUNDS
// BOUNDS: destination K window exceeds the source channel/filter extent
// RUN: sed 's/%c8 = arith.constant 8 : index/%c8 = arith.constant 65536 : index/' %S/img2col-l1-to-l0a-half.mlir > %t.fmap-range.mlir
// RUN: not %tla_compile %t.fmap-range.mlir -o - 2>&1 | %filecheck %s --check-prefix=FMAP-RANGE
// FMAP-RANGE: fmap_h must be in 1..65535
// RUN: sed 's/%c1 = arith.constant 1 : index/%c1 = arith.constant 0 : index/' %S/img2col-l1-to-l0a-half.mlir > %t.stride-range.mlir
// RUN: not %tla_compile %t.stride-range.mlir -o - 2>&1 | %filecheck %s --check-prefix=STRIDE-RANGE
// STRIDE-RANGE: stride_h must be in 1..255
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
// DST-LAYOUT: img2col destination must be tagged zN
