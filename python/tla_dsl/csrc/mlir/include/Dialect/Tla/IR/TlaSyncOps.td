//===----------------------------------------------------------------------===//
// Synchronization ops: flags, barriers and mutexes.
//
// Included by Dialect/Tla/IR/Tla.td, the file mlir-tblgen is run on.
//===----------------------------------------------------------------------===//

def Tla_CrossCoreSetFlagOp : Tla_Op<"cross_core_set_flag", []> {
  let summary = "Cross-core flag set";
  let description = [{
    Set a cross-core synchronization flag.
  }];
  let arguments = (ins Tla_CrossFlagType:$flag, Tla_PipeAttr:$pipe,
                       OptionalAttr<I64Attr>:$aiv_id);
  let assemblyFormat = [{
    $flag attr-dict `:` qualified(type($flag))
  }];
  let hasVerifier = 1;
}

def Tla_CrossCoreWaitFlagOp : Tla_Op<"cross_core_wait_flag", []> {
  let summary = "Cross-core flag wait";
  let description = [{
    Wait on a cross-core synchronization flag.
  }];
  let arguments = (ins Tla_CrossFlagType:$flag, Tla_PipeAttr:$pipe,
                       OptionalAttr<I64Attr>:$aiv_id);
  let assemblyFormat = [{
    $flag attr-dict `:` qualified(type($flag))
  }];
  let hasVerifier = 1;
}

def Tla_CrossFlagOp : Tla_Op<"cross_flag", [Pure]> {
  let summary = "Cross-block flag literal";
  let description = [{
    Materialize a cross-core synchronization flag.
  }];
  let arguments = (ins StrAttr:$name);
  let results = (outs Tla_CrossFlagType:$flag);
  let assemblyFormat = [{
    $name attr-dict `->` qualified(type($flag))
  }];
}

def Tla_FlagOp : Tla_Op<"flag", [Pure]> {
  let summary = "Flag literal";
  let description = [{
    Materialize a static synchronization flag.
    ``id = -1`` requests automatic allocation; ``0`` through ``7`` select an
    explicit hardware event id for the source/destination pipe pair.
    Automatic allocation skips explicit ids. Explicit reuse is allowed when
    the caller ensures synchronization lifetimes do not overlap.
  }];
  let arguments = (ins StrAttr:$name, Tla_PipeAttr:$src_pipe, Tla_PipeAttr:$dst_pipe,
                       DefaultValuedAttr<I64Attr, "-1">:$id);
  let results = (outs Tla_FlagType:$flag);
  let assemblyFormat = [{
    $name attr-dict `->` qualified(type($flag))
  }];
  let hasVerifier = 1;
}

def Tla_PipeBarrierOp : Tla_Op<"pipe_barrier", []> {
  let summary = "Pipe barrier";
  let description = [{
    Apply a synchronization barrier to the specified pipe.
  }];
  let arguments = (ins Tla_PipeAttr:$pipe);
  let assemblyFormat = [{
    `[` $pipe `]` attr-dict
  }];
  let hasVerifier = 1;
}

def Tla_MutexOp : Tla_Op<"mutex", [Pure]> {
  let summary = "Mutex literal";
  let description = [{
    Materialize a static mutex associated with a semantic resource.
    ``id = -1`` means the compiler may allocate the hardware MutexID later;
    ``0`` through ``31`` are user-provided hardware MutexIDs.
  }];
  let arguments = (ins StrAttr:$resource, I64Attr:$id);
  let results = (outs Tla_MutexType:$mutex);
  let assemblyFormat = [{
    $resource attr-dict `->` qualified(type($mutex))
  }];
}

def Tla_MutexLockOp : Tla_Op<"mutex_lock", []> {
  let summary = "Mutex lock";
  let description = [{
    Acquire a mutex from the specified pipe.
  }];
  let arguments = (ins Tla_MutexType:$mutex, Tla_PipeAttr:$pipe);
  let assemblyFormat = [{
    $mutex `[` $pipe `]` `:` qualified(type($mutex)) attr-dict
  }];
  let hasVerifier = 1;
}

def Tla_MutexUnlockOp : Tla_Op<"mutex_unlock", []> {
  let summary = "Mutex unlock";
  let description = [{
    Release a mutex from the specified pipe.
  }];
  let arguments = (ins Tla_MutexType:$mutex, Tla_PipeAttr:$pipe);
  let assemblyFormat = [{
    $mutex `[` $pipe `]` `:` qualified(type($mutex)) attr-dict
  }];
  let hasVerifier = 1;
}

def Tla_LocalMemBarOp : Tla_Op<"local_mem_bar", []> {
  let summary = "Local memory barrier";
  let description = [{
    Apply a synchronization barrier to the specified memory type.
    The ``barrier_kind`` is an encoded I32 immediate that selects the
    src/dst pair (see hivmave.membar for the encoding scheme).
  }];
  let arguments = (ins I64Attr:$barrier_kind);
  let assemblyFormat = "$barrier_kind attr-dict";
  let hasVerifier = 1;
}

def Tla_DataSyncBarrierOp : Tla_Op<"data_sync_barrier", []> {
  let summary = "Data synchronization barrier over a storage domain (ISA dsb)";
  let description = [{
    Block subsequent instructions until the memory access instructions of the
    given storage domain that were issued earlier have completed. The domain
    is one of ``gm_ub`` (GM and UB), ``gm`` (GM), ``ub`` (UB) or ``seq``
    (sequencer, including the parameter buffer).

    This is one ``dsb`` instruction. The HIVM dialect has no op for it and the
    CONV-style route via ``llvm.call_intrinsic`` is closed to hivmc, so
    ``tla-lower-scalar-ops`` rewrites this op into a call to an always_inline
    meta-op bitcode helper that calls ``AscendC::DataSyncBarrier<MemDsbT>``;
    ccec selects ``llvm.hivm.DSB`` there and inlining leaves a single
    instruction at the call site. The ``gm_ub``/``gm``/``seq`` helpers live in
    bc/Scalar/ under one shared name -- their bodies are identical per core.
    The ``ub`` helper is core-specific (``_mlir_ciface_tla_data_sync_barrier_
    ub_aiv`` in bc/Vector/): the AIC-compiled body would be empty, and hivmc
    links both meta-op bitcodes into each subcore compile, so a shared name
    could let the empty AIC body win on AIV and silently drop the barrier
    (same hazard as ``_mlir_ciface_tla_init_soc_state``). AIC functions drop
    the op entirely.

    Not nested in ``tla.vec.func``: ``dsb`` is a scalar-pipeline instruction
    over whole storage domains, not a per-thread view like the VF membar
    (tla.local_mem_bar), and it has no vec.func form. The ``ub`` domain is
    rejected inside ``tla.cube`` at parse time -- AIC has no UB and would not
    issue the instruction, so the barrier would sit there silently doing
    nothing. ``gm_ub`` covers GM and UB only: L1 and the L0A/L0B/L0C buffers are
    NOT waited on; order those with ``tla.pipe_barrier`` (MTE1/FIX/CUBE) or
    ``tla.flag`` set/wait pairs. In a mixed kernel's scalar body -- duplicated
    per core -- the ``ub`` domain is a no-op on the AIC copy.
  }];
  let arguments = (ins Tla_DsbTargetAttr:$domain);
  let assemblyFormat = "$domain attr-dict";
  let hasVerifier = 1;
}

def Tla_SetFlagOp : Tla_Op<"set_flag", []> {
  let summary = "Set flag";
  let description = [{
    Signal a synchronization flag from one pipe to another.
  }];
  let arguments = (ins Tla_FlagType:$flag);
  let assemblyFormat = [{
    $flag `:` qualified(type($flag)) attr-dict
  }];
  let hasVerifier = 1;
}

def Tla_WaitFlagOp : Tla_Op<"wait_flag", []> {
  let summary = "Wait flag";
  let description = [{
    Wait on a synchronization flag from one pipe to another.
  }];
  let arguments = (ins Tla_FlagType:$flag);
  let assemblyFormat = [{
    $flag `:` qualified(type($flag)) attr-dict
  }];
  let hasVerifier = 1;
}
