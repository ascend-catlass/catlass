// RUN: %tla_compile %s --mlir-print-ir-after=tla-vector-region-simt -o %t 2>&1 | %filecheck %s

// A per-thread atomic add lowers to llvm.hivm.atom.ADD.<G|S>.<type> at the
// address the caller computed. The pointer carries the element type and address
// space: gm picks G (1), ub picks S (6), and the pointee picks the type suffix.
// Unsigned integers reuse the signed form -- addition is bit-identical -- with a
// bitcast to the signless storage type the LLVM dialect carries.
//
// A !tla.ptr cannot cross the launch boundary, so the chain is rebuilt inside the
// outlined function from its root: a captured tensor (tensor_ptr) or a captured
// integer address (inttoptr), plus one byte offset per ptr_add between the root
// and the atomic.

!gm_i32 = !tla.tensor<!tla.layout<!tla.shape<16>, !tla.stride<1>, !tla.shape<16>, RowMajor>, !tla.coord<0>, !tla.ptr<i32, gm, 4>>
!gm_ui32 = !tla.tensor<!tla.layout<!tla.shape<16>, !tla.stride<1>, !tla.shape<16>, RowMajor>, !tla.coord<0>, !tla.ptr<ui32, gm, 4>>
!gm_i64 = !tla.tensor<!tla.layout<!tla.shape<16>, !tla.stride<1>, !tla.shape<16>, RowMajor>, !tla.coord<0>, !tla.ptr<i64, gm, 8>>
!gm_ui64 = !tla.tensor<!tla.layout<!tla.shape<16>, !tla.stride<1>, !tla.shape<16>, RowMajor>, !tla.coord<0>, !tla.ptr<ui64, gm, 8>>
!gm_f16 = !tla.tensor<!tla.layout<!tla.shape<16>, !tla.stride<1>, !tla.shape<16>, RowMajor>, !tla.coord<0>, !tla.ptr<f16, gm, 2>>
!ub_bf16 = !tla.tensor<!tla.layout<!tla.shape<16>, !tla.stride<1>, !tla.shape<16>, RowMajor>, !tla.coord<0>, !tla.ptr<bf16, ub, 2>>

module {
  func.func @simt_atomic_add(%gm_i32_memref: memref<16xi32, #hivm.address_space<gm>>,
                             %gm_ui32_memref: memref<16xui32, #hivm.address_space<gm>>,
                             %gm_i64_memref: memref<16xi64, #hivm.address_space<gm>>,
                             %gm_ui64_memref: memref<16xui64, #hivm.address_space<gm>>,
                             %gm_f16_memref: memref<16xf16, #hivm.address_space<gm>>,
                             %ub_bf16_memref: memref<16xbf16, #hivm.address_space<ub>>,
                             %raw_addr: i64) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c16 = arith.constant 16 : index
    %gm_i32 = tla.tensor_desc %gm_i32_memref shape [%c16, %c1, %c1, %c1] stride [%c1, %c1, %c1, %c1] origin_shape [%c16, %c1] coord [%c0, %c0] : memref<16xi32, #hivm.address_space<gm>> -> !gm_i32
    %gm_ui32 = tla.tensor_desc %gm_ui32_memref shape [%c16, %c1, %c1, %c1] stride [%c1, %c1, %c1, %c1] origin_shape [%c16, %c1] coord [%c0, %c0] : memref<16xui32, #hivm.address_space<gm>> -> !gm_ui32
    %gm_i64 = tla.tensor_desc %gm_i64_memref shape [%c16, %c1, %c1, %c1] stride [%c1, %c1, %c1, %c1] origin_shape [%c16, %c1] coord [%c0, %c0] : memref<16xi64, #hivm.address_space<gm>> -> !gm_i64
    %gm_ui64 = tla.tensor_desc %gm_ui64_memref shape [%c16, %c1, %c1, %c1] stride [%c1, %c1, %c1, %c1] origin_shape [%c16, %c1] coord [%c0, %c0] : memref<16xui64, #hivm.address_space<gm>> -> !gm_ui64
    %gm_f16 = tla.tensor_desc %gm_f16_memref shape [%c16, %c1, %c1, %c1] stride [%c1, %c1, %c1, %c1] origin_shape [%c16, %c1] coord [%c0, %c0] : memref<16xf16, #hivm.address_space<gm>> -> !gm_f16
    %ub_bf16 = tla.tensor_desc %ub_bf16_memref shape [%c16, %c1, %c1, %c1] stride [%c1, %c1, %c1, %c1] origin_shape [%c16, %c1] coord [%c0, %c0] : memref<16xbf16, #hivm.address_space<ub>> -> !ub_bf16
    // An alias only re-shapes a pointer, so a pointer taken from it has to reach
    // the tensor behind the chain -- capturing the alias would drop its address.
    %alias_ptr = tla.tensor_ptr %gm_i32 : !gm_i32 -> !tla.ptr<i32, gm, 4>
    %alias = tla.make_tensor_like %alias_ptr like %gm_i32 layoutTag(<RowMajor>) : !tla.ptr<i32, gm, 4>, !gm_i32 -> !gm_i32
    "tla.vector"() ({
      "tla.vec.func"() ({
        %i = arith.constant 3 : index
        %p_i32 = tla.tensor_ptr %gm_i32 : !gm_i32 -> !tla.ptr<i32, gm, 4>
        %q_i32 = tla.ptr_add %p_i32, %i : !tla.ptr<i32, gm, 4>, index -> !tla.ptr<i32, gm, 4>
        %one_i32 = arith.constant 1 : i32
        %old_i32 = tla.simt_atomic_add %q_i32, %one_i32 : !tla.ptr<i32, gm, 4>, i32 -> i32
        %q_i32_2 = tla.ptr_add %p_i32, %i : !tla.ptr<i32, gm, 4>, index -> !tla.ptr<i32, gm, 4>
        %old2_i32 = tla.simt_atomic_add %q_i32_2, %old_i32 : !tla.ptr<i32, gm, 4>, i32 -> i32
        %p_ui32 = tla.tensor_ptr %gm_ui32 : !gm_ui32 -> !tla.ptr<ui32, gm, 4>
        %q_ui32 = tla.ptr_add %p_ui32, %i : !tla.ptr<ui32, gm, 4>, index -> !tla.ptr<ui32, gm, 4>
        %one_ui32_bits = arith.constant 1 : i32
        %one_ui32 = builtin.unrealized_conversion_cast %one_ui32_bits : i32 to ui32
        %old_ui32 = tla.simt_atomic_add %q_ui32, %one_ui32 : !tla.ptr<ui32, gm, 4>, ui32 -> ui32
        // The old value feeds a same-type store: the intrinsic returns i32, so
        // the lowering must bridge it back to ui32.
        tla.simt_store %gm_ui32[%c0], %old_ui32 : <!tla.layout<!tla.shape<16>, !tla.stride<1>, !tla.shape<16>, RowMajor>, !tla.coord<0>, !tla.ptr<ui32, gm, 4>>, ui32
        %p_i64 = tla.tensor_ptr %gm_i64 : !gm_i64 -> !tla.ptr<i64, gm, 8>
        %one_i64 = arith.constant 1 : i64
        tla.simt_atomic_add %p_i64, %one_i64 : !tla.ptr<i64, gm, 8>, i64 -> i64
        %p_ui64 = tla.tensor_ptr %gm_ui64 : !gm_ui64 -> !tla.ptr<ui64, gm, 8>
        %one_ui64_bits = arith.constant 1 : i64
        %one_ui64 = builtin.unrealized_conversion_cast %one_ui64_bits : i64 to ui64
        %old_ui64 = tla.simt_atomic_add %p_ui64, %one_ui64 : !tla.ptr<ui64, gm, 8>, ui64 -> ui64
        tla.simt_store %gm_ui64[%c0], %old_ui64 : <!tla.layout<!tla.shape<16>, !tla.stride<1>, !tla.shape<16>, RowMajor>, !tla.coord<0>, !tla.ptr<ui64, gm, 8>>, ui64
        %p_f16 = tla.tensor_ptr %gm_f16 : !gm_f16 -> !tla.ptr<f16, gm, 2>
        %one_f16 = arith.constant 1.000000e+00 : f16
        tla.simt_atomic_add %p_f16, %one_f16 : !tla.ptr<f16, gm, 2>, f16 -> f16
        %p_bf16 = tla.tensor_ptr %ub_bf16 : !ub_bf16 -> !tla.ptr<bf16, ub, 2>
        %one_bf16 = arith.constant 1.000000e+00 : bf16
        tla.simt_atomic_add %p_bf16, %one_bf16 : !tla.ptr<bf16, ub, 2>, bf16 -> bf16
        // A pointer that does not come from a tensor keeps its integer root.
        %raw = tla.inttoptr %raw_addr : i64 -> !tla.ptr<i32, gm, 4>
        %raw_i = tla.ptr_add %raw, %i : !tla.ptr<i32, gm, 4>, index -> !tla.ptr<i32, gm, 4>
        tla.simt_atomic_add %raw_i, %one_i32 : !tla.ptr<i32, gm, 4>, i32 -> i32
        // A recast changes the element unit mid-chain: the f32 ptr_add after it
        // is scaled by 4 bytes, the i32 ptr_add before it by 4 bytes of its own.
        %p_f32 = tla.recast_ptr %p_i32 : !tla.ptr<i32, gm, 4> -> !tla.ptr<f32, gm, 4>
        %q_f32 = tla.ptr_add %p_f32, %i : !tla.ptr<f32, gm, 4>, index -> !tla.ptr<f32, gm, 4>
        %one_f32 = arith.constant 1.000000e+00 : f32
        tla.simt_atomic_add %q_f32, %one_f32 : !tla.ptr<f32, gm, 4>, f32 -> f32
        // An alias of the same storage resolves to the tensor it was built from.
        %alias_atomic_ptr = tla.tensor_ptr %alias : !gm_i32 -> !tla.ptr<i32, gm, 4>
        tla.simt_atomic_add %alias_atomic_ptr, %one_i32 : !tla.ptr<i32, gm, 4>, i32 -> i32
      }) {mode = "simt", thread_block_dim = array<i64: 64, 1, 1>} : () -> ()
    }) : () -> ()
    return
  }
}

// gm integer: G.s32 on the gm pointer; the old value rides the intrinsic
// result, and the chained second add consumes it as its addend. The tensor's
// address arrives as a memref parameter, and the element offset becomes a byte
// GEP.
// CHECK: memref.extract_aligned_pointer_as_index %{{.*}} : memref<16xi32, 1>
// CHECK: llvm.hivm.atom.ADD.G.s32
// CHECK-SAME: !llvm.ptr<1>
// CHECK: llvm.hivm.atom.ADD.G.s32

// ui32 is the bit pattern of i32: signed intrinsic, with the value carried in
// the signless integer type the LLVM dialect uses. The i32 result is bridged
// back to ui32 before the same-type store.
// CHECK: builtin.unrealized_conversion_cast %{{.*}} : ui32 to i32
// CHECK: llvm.hivm.atom.ADD.G.s32
// CHECK: builtin.unrealized_conversion_cast %{{.*}} : i32 to ui32

// CHECK: llvm.hivm.atom.ADD.G.s64
// CHECK: builtin.unrealized_conversion_cast %{{.*}} : ui64 to i64
// CHECK: llvm.hivm.atom.ADD.G.s64
// CHECK: builtin.unrealized_conversion_cast %{{.*}} : i64 to ui64
// CHECK: llvm.hivm.atom.ADD.G.fp16

// ub bf16: S address space 6, bf16 suffix.
// CHECK: llvm.hivm.atom.ADD.S.bf16
// CHECK-SAME: !llvm.ptr<6>

// A raw integer address is not captured as a tensor: the outlined function
// receives the i64 and rebuilds the pointer itself.
// CHECK: llvm.inttoptr %{{.*}} : i64 to !llvm.ptr<1>
// CHECK: llvm.hivm.atom.ADD.G.s32

// A recast mid-chain keeps its own element unit: the f32 atomic still reaches
// G.fp32 and the offsets stay byte offsets.
// CHECK: llvm.hivm.atom.ADD.G.fp32

// The alias capture resolves to the tensor it was built from, so the atomic
// adds on the same storage as the direct tensor pointer.
// CHECK: llvm.hivm.atom.ADD.G.s32

// CHECK-NOT: tla.simt_atomic_add
// CHECK-NOT: tla.tensor_ptr
// CHECK-NOT: tla.ptr_add
// CHECK-NOT: tla.recast_ptr
