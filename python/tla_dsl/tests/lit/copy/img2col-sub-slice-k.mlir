// A destination K window only has to be C0-aligned: it may lie wholly inside one
// C1 slice (kh * kw * C0 = 144 elements for the 3x3 filter here) and it may
// straddle a slice boundary. Load3Dv2 takes kStartPt/kExtension verbatim from
// the destination tile and reads channelSize from the source, so a partial
// window still describes the same feature map.

// Extent 48 starting at 16: inside the first C1 slice, on neither end aligned
// to a slice boundary.
// RUN: sed 's/(16,18)/(16,3)/g; s/64,288/64,48/g; s/zN>, !tla.coord<0,0>, !tla.ptr<f16, l0a/zN>, !tla.coord<0,16>, !tla.ptr<f16, l0a/g' %S/img2col-l1-to-l0a-half.mlir > %t.inside.mlir
// RUN: %tla_compile %t.inside.mlir -o - | %filecheck %s

// Extent 96 starting at 96: spans the 144-element boundary between the two C1
// slices -- the window aclnn's own L0 tiling produces for kAL1 = 864.
// RUN: sed 's/(16,18)/(16,6)/g; s/64,288/64,96/g; s/zN>, !tla.coord<0,0>, !tla.ptr<f16, l0a/zN>, !tla.coord<0,96>, !tla.ptr<f16, l0a/g' %S/img2col-l1-to-l0a-half.mlir > %t.straddle.mlir
// RUN: %tla_compile %t.straddle.mlir -o - | %filecheck %s

// CHECK: call @img2col_l1_zNUnAlign_to_l0a_zN_half
// CHECK-NOT: tla.img2col
