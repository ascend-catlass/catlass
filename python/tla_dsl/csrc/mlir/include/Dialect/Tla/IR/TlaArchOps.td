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

// InitSocState is NOT a user-facing TLA op. ``tla-prologue-epilogue`` injects
// ``func.call @_mlir_ciface_tla_init_soc_state`` (Scalar BC → AscendC::InitSocState)
// at every HACC entry (InitSoc only; no extra CTRL[60]/[48]/[56]). No public API.
