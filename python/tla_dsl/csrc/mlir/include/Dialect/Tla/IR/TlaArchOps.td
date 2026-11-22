//===----------------------------------------------------------------------===//
// Architecture queries: the ``arch.*`` ops that report where the running code
// sits in the grid, and the SIMT thread facilities.
//
// These are not tied to one execution mode, which is why they are not in
// TlaSimdOps.td or TlaSimtOps.td. Each group below says where it applies.
//
// Included by Dialect/Tla/IR/Tla.td, the file mlir-tblgen is run on.
//===----------------------------------------------------------------------===//

// Deliberately not Pure: these controls must survive dead-code elimination.
def Tla_ProfStartOp : Tla_Op<"prof_start", []> {
  let summary = "Start best-effort device profiling at this position";
  let assemblyFormat = [{ attr-dict }];
}

def Tla_ProfEndOp : Tla_Op<"prof_end", []> {
  let summary = "End best-effort device profiling at this position";
  let assemblyFormat = [{ attr-dict }];
}

// Grid coordinates. Used by every kernel -- SIMD, SIMT, cube and mixed alike.
def Tla_BlockNumOp : Tla_Op<"arch.block_num", [Pure]> {
  let summary = "Block count";
  let description = [{
    Return the number of blocks (AI cores) in the launch as signless ``i32``.
    This is the grid extent, matching AscendC ``GetBlockNum()``; the per-block
    thread extents are ``tla.arch.thread_block_dim``.
  }];
  let results = (outs I32:$num);
  let assemblyFormat = [{
    attr-dict `->` type($num)
  }];
}

def Tla_BlockIdxOp : Tla_Op<"arch.block_idx", [Pure]> {
  let summary = "Block index";
  let description = [{
    Return the current block index as signless ``i32``.
  }];
  let results = (outs I32:$index);
  let assemblyFormat = [{
    attr-dict `->` type($index)
  }];
}

// Which AIV sub-block of the pair this code runs on. Vector and mixed kernels;
// on a cube-only kernel there is no sub-block to name.
def Tla_SubBlockIdxOp : Tla_Op<"arch.sub_block_idx", [Pure]> {
  let summary = "Sub-block index";
  let description = [{
    Return the current sub-block index as signless ``i32``.
  }];
  let results = (outs I32:$index);
  let assemblyFormat = [{
    attr-dict `->` type($index)
  }];
}

// Thread coordinates. SIMT only: meaningful inside a ``tla.vec.func`` with
// ``mode = "simt"``.
def Tla_ArchLaneIdxOp : Tla_Op<"arch.lane_idx", [Pure]> {
  let summary = "SIMT lane index within the current warp";
  let description = [{
    Return the x-fastest linear thread index modulo 32 as signless ``i32``.
    Only available inside a ``tla.vec.func`` with ``mode = "simt"``.
  }];
  let results = (outs I32:$result);
  let hasVerifier = 1;
  let assemblyFormat = [{ attr-dict `->` type($result) }];
}

def Tla_ArchLanemaskOp : Tla_Op<"arch.lanemask", [Pure]> {
  let summary = "SIMT positional lane mask independent of thread activity";
  let arguments = (ins StrAttr:$mode);
  let results = (outs I32:$result);
  let hasVerifier = 1;
  let assemblyFormat = "$mode attr-dict `:` type($result)";
}

// Activity-dependent: unknown effects also prevent CSE across active sets.
// NoMemoryEffect alone would permit a dominating query to replace a branch query.
def Tla_ArchActivemaskOp : Tla_Op<"arch.activemask", []> {
  let summary = "Mask of lanes active at this SIMT instruction";
  let description = [{
    Return a bit for each currently active lane in the calling warp. Inactive
    lanes, including unlaunched lanes of a partial warp, contribute no bits.
    This query is convergent and must remain at its original control-flow site.
  }];
  let results = (outs I32:$result);
  let hasVerifier = 1;
  let assemblyFormat = [{ attr-dict `->` type($result) }];
}

def Tla_ThreadIdxOp : Tla_Op<"arch.thread_idx", [Pure]> {
  let summary = "SIMT thread index";
  let description = [{
    Return the current thread's ``(x, y, z)`` index inside its thread block as
    signless ``i32``. Only meaningful inside a ``tla.vec.func`` with
    ``mode = "simt"``; it lowers to the three separate
    ``hivm_regbaseintrins.thread_id_{x,y,z}`` intrinsics. Results with no uses
    are not materialized.
  }];
  let results = (outs I32:$x, I32:$y, I32:$z);
  let assemblyFormat = [{
    attr-dict `->` type($x) `,` type($y) `,` type($z)
  }];
}

def Tla_ThreadBlockDimOp : Tla_Op<"arch.thread_block_dim", [Pure]> {
  let summary = "SIMT thread-block dimensions";
  let description = [{
    Return the enclosing thread block's ``(x, y, z)`` extents as signless
    ``i32``. This is the SIMT thread geometry; the number of blocks in the
    launch is ``tla.arch.block_num``. Lowers to the three separate
    ``hivm_regbaseintrins.block_dim_{x,y,z}`` intrinsics. Results with no uses
    are not materialized.
  }];
  let results = (outs I32:$x, I32:$y, I32:$z);
  let assemblyFormat = [{
    attr-dict `->` type($x) `,` type($y) `,` type($z)
  }];
}

// Barrier across the threads of a block. SIMT only.
def Tla_SyncThreadsOp : Tla_Op<"arch.sync_threads", []> {
  let summary = "SIMT block-wide thread barrier";
  let description = [{
    Synchronize every thread of the enclosing ``tla.vec.func`` with
    ``mode = "simt"``. Required between a write and a read of shared data by
    different threads. Lowers to ``hivm_regbaseintrins.sync_threads``.
  }];
  let assemblyFormat = [{ attr-dict }];
}

// Intra-warp register exchange. SIMT only.
def Tla_ArchShuffleOp : Tla_Op<"arch.shuffle",
    [NoMemoryEffect, AllTypesMatch<["value", "result"]>]> {
  let summary = "SIMT intra-warp register exchange";
  let description = [{
    Return the ``value`` held by another thread of the same warp, without going
    through memory. ``mode`` says which lane is read: ``idx`` reads lane
    ``index`` of the group, ``up`` reads ``laneid - index``, ``down`` reads
    ``laneid + index``, and ``bfly`` reads ``laneid ^ index``.

    ``width`` splits the warp into contiguous groups of that many lanes; it must
    be a power of two in ``[2, 32]``. It does not change the set of threads that
    execute the op. Each mode bounds only the one edge its arithmetic can cross:
    ``up`` moves downward and checks the group's first lane, ``down`` moves
    upward and checks its last, and a lane whose source falls outside in the
    checked direction keeps its own value. ``bfly`` can cross either edge but is
    checked against the last lane only, so a mask of ``width`` or more reads a
    lower-numbered group rather than keeping the lane's own value. Confining a
    butterfly to one group is the caller's choice of ``index``, not a property
    this op enforces.

    ``index`` travels in a 5-bit control field, so it is carried exactly up to
    31 and wraps from 32 -- 33 would become 1 and read a neighbour. For ``up``,
    ``down`` and ``bfly`` the index is a distance or a mask, so a constant of 32
    or more is rejected; ``idx`` is exempt, since there it is a lane number and
    the masking is exactly the ``index % width`` this op documents. An index
    below 32 that sends a lane outside its group is a legal request, answered by
    leaving that lane its own value.

    ``value`` is signless because signedness is not a property of a bit pattern
    being moved: the frontend's unsigned types are shuffled as signless integers
    of the same width and carry their Python type through to the result.

    Lowers to the ``hivm_regbaseintrins.intr.hivm.shfl.{idx,up,down,bfly}.*``
    intrinsic for the value type. ``i32``, ``i64``, ``f32`` and ``f16`` have a
    native intrinsic; narrower integers and ``bf16`` are widened to ``i32`` and
    truncated back, which is exact because a shuffle moves bit patterns and does
    not interpret them.

    Deliberately not ``Pure``: the result depends on which threads reach the op,
    so hoisting it out of a branch or loop would change its meaning.
  }];
  let arguments = (ins
    AnyTypeOf<[I8, I16, I32, I64, F16, F32, BF16]>:$value,
    I32:$index,
    Tla_ShuffleModeAttr:$mode,
    DefaultValuedAttr<I32Attr, "32">:$width
  );
  let results = (outs AnyTypeOf<[I8, I16, I32, I64, F16, F32, BF16]>:$result);
  let hasVerifier = 1;
  let assemblyFormat = [{
    $mode $value `,` $index attr-dict `:` type($value)
  }];
}

// InitSocState is NOT a user-facing TLA op. ``tla-prologue-epilogue`` injects
// ``func.call @_mlir_ciface_tla_init_soc_state`` (Scalar BC → AscendC::InitSocState)
// at every HACC entry (InitSoc only; no extra CTRL[60]/[48]/[56]). No public API.

// Active-set dependent: leave effects unknown to prevent CSE across divergent
// control flow as well as speculation. NoMemoryEffect alone would allow CSE.
def Tla_ArchWarpReduceOp : Tla_Op<"arch.warp_reduce",
    [AllTypesMatch<["value", "result"]>]> {
  let summary = "SIMT reduction over the active lanes of a warp";
  let arguments = (ins Tla_ScalarValueType:$value, StrAttr:$kind, UnitAttr:$isUnsigned);
  let results = (outs Tla_ScalarValueType:$result);
  let hasVerifier = 1;
  let assemblyFormat = "$kind $value attr-dict `:` type($result)";
}
