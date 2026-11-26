//===----------------------------------------------------------------------===//
// Per-thread (SIMT) ops: the simt_* family run inside a ``tla.vec.func``
// with ``mode = "simt"``. The arch.* queries they use live in TlaArchOps.td.
//
// Included by Dialect/Tla/IR/Tla.td, the file mlir-tblgen is run on.
//===----------------------------------------------------------------------===//

// Per-thread scalar arithmetic inside a ``tla.vec.func`` with ``mode = "simt"``.
// These are NOT the tla.add / tla.sub / ... family: those are whole-vector AVE
// operations over ``!tla.vector`` SSA carrying lane masks. Each of these is one
// thread combining one element with another, and lowers to the matching arith op
// for the element type.
class Tla_SimtBinaryOp<string mnemonic, string opSummary, string lowering>
    : Tla_Op<mnemonic, [Pure, AllTypesMatch<["lhs", "rhs", "result"]>]> {
  let summary = opSummary;
  let description = [{
    Per-thread scalar operation inside a ``tla.vec.func`` with
    ``mode = "simt"``. Lowers to }] # lowering # [{ on the element type.
  }];
  let arguments = (ins Tla_ScalarValueType:$lhs, Tla_ScalarValueType:$rhs);
  let results = (outs Tla_ScalarValueType:$result);
  let assemblyFormat = [{
    $lhs `,` $rhs attr-dict `:` type($result)
  }];
}

def Tla_SimtAddOp : Tla_SimtBinaryOp<"simt_add", "Per-thread scalar add",
    "``arith.addf`` / ``arith.addi``">;
def Tla_SimtSubOp : Tla_SimtBinaryOp<"simt_sub", "Per-thread scalar subtract",
    "``arith.subf`` / ``arith.subi``">;
def Tla_SimtMulOp : Tla_SimtBinaryOp<"simt_mul", "Per-thread scalar multiply",
    "``arith.mulf`` / ``arith.muli``">;
def Tla_SimtDivOp : Tla_SimtBinaryOp<"simt_div", "Per-thread scalar divide",
    "``arith.divf`` / ``arith.divsi``">;
def Tla_SimtPowOp : Tla_SimtBinaryOp<"simt_pow", "Per-thread scalar power",
    "``math.powf``">;

// Per-thread minimum/maximum. These are the NaN-ignoring fmin/fmax forms: given
// one NaN operand they return the other, and only both-NaN yields NaN. The
// NaN-propagating IEEE-754-2019 minimum/maximum are not available -- the
// backend cannot select `fmaximum`/`fminimum` for f32, f16 or bf16 -- so the
// result differs from torch.maximum only when an input is NaN.
def Tla_SimtMaxOp : Tla_SimtBinaryOp<"simt_max", "Per-thread scalar maximum",
    "``llvm.intr.maxnum`` for floats and ``llvm.intr.smax`` for integers">;
def Tla_SimtMinOp : Tla_SimtBinaryOp<"simt_min", "Per-thread scalar minimum",
    "``llvm.intr.minnum`` for floats and ``llvm.intr.smin`` for integers">;

def Tla_SimtCastOp : Tla_Op<"simt_cast", [Pure]> {
  let summary = "Per-thread scalar element-type conversion";
  let description = [{
    Convert a per-thread scalar to another element type inside a
    ``tla.vec.func`` with ``mode = "simt"``.

    Unlike ``tla.cast``, which converts a whole VectorSSA and carries the AVE
    cast knobs (register slot, saturation, rounding), this is a single scalar
    conversion with the target's fixed behaviour: ``truncf`` rounds to nearest
    even and ``fptosi`` truncates toward zero.

    ``kind`` names the conversion -- ``extsi``, ``extui``, ``trunci``,
    ``extf``, ``truncf``, ``sitofp``, ``uitofp``, ``fptosi``, ``fptoui`` --
    because MLIR integers are signless and the source type's signedness would
    otherwise be lost. It lowers to the ``arith`` op of the same name.
  }];
  let arguments = (ins Tla_ScalarValueType:$source, StrAttr:$kind);
  let results = (outs Tla_ScalarValueType:$result);
  let hasVerifier = 1;
  let assemblyFormat = [{
    $kind $source attr-dict `:` type($source) `to` type($result)
  }];
}

def Tla_SimtWhereOp : Tla_Op<"simt_where",
    [Pure, AllTypesMatch<["x", "y", "result"]>]> {
  let summary = "Per-thread scalar select";
  let description = [{
    Per-thread select inside a ``tla.vec.func`` with ``mode = "simt"``: the
    result is ``x`` when the condition holds and ``y`` otherwise.

    Unlike ``tla.where``, which selects per lane under a ``!tla.mask_ssa``, the
    condition here is the plain ``i1`` that ``tla.simt_cmp`` produces, so this
    is the per-thread form of ``x if cond else y``.

    Lowers to ``arith.select``.
  }];
  let arguments = (ins I1:$condition, Tla_ScalarValueType:$x,
                       Tla_ScalarValueType:$y);
  let results = (outs Tla_ScalarValueType:$result);
  let assemblyFormat = [{
    $condition `,` $x `,` $y attr-dict `:` type($result)
  }];
}

def Tla_SimtCmpOp : Tla_Op<"simt_cmp", [Pure, AllTypesMatch<["lhs", "rhs"]>]> {
  let summary = "Per-thread scalar comparison";
  let description = [{
    Compare two per-thread scalars inside a ``tla.vec.func`` with
    ``mode = "simt"``. ``mode`` is one of ``lt``, ``le``, ``gt``, ``ge``,
    ``eq``, ``ne``.

    Unlike ``tla.cmp``, which produces a per-lane ``!tla.mask_ssa`` for
    predicating whole-vector ops, this yields a plain ``i1`` for one thread, so
    it feeds ordinary control flow and selects.

    Lowers to ``arith.cmpf`` (ordered) for floats and ``arith.cmpi`` for
    integers. MLIR integers are signless, so ``isUnsigned`` records which
    comparison the source type asked for; it is ignored for floats and for
    ``eq``/``ne``.
  }];
  let arguments = (ins Tla_ScalarValueType:$lhs, Tla_ScalarValueType:$rhs,
                       StrAttr:$mode, UnitAttr:$isUnsigned);
  let results = (outs I1:$result);
  let hasVerifier = 1;
  let assemblyFormat = [{
    $mode $lhs `,` $rhs attr-dict `:` type($lhs)
  }];
}

// Per-thread scalar math. Each lowers to the matching `math` op, which reaches
// the hardware through --convert-math-to-llvm. Only the subset that is
// measured to work inside a SIMT vector function is exposed here. sin/cos,
// floor/ceil/round, log2/exp2, tanh/erf/atan have no instruction on this target
// and need a bitcode helper instead. log takes a different route: see
// Tla_SimtLogOp below.
//
// bf16 has no transcendental unit -- exp, log, sqrt and pow all fail to
// select -- so for a bf16 operand tla-vector-region-simt evaluates these in f32 and
// rounds the result back. The f32 intermediate makes them slightly more
// accurate than a native bf16 evaluation would be. simt_abs needs no such
// promotion; it has a bf16 instruction.
class Tla_SimtUnaryOp<string mnemonic, string opSummary, string lowering>
    : Tla_Op<mnemonic, [Pure, AllTypesMatch<["operand", "result"]>]> {
  let summary = opSummary;
  let description = [{
    Per-thread scalar operation inside a ``tla.vec.func`` with
    ``mode = "simt"``. Lowers to }] # lowering # [{.
  }];
  let arguments = (ins Tla_ScalarValueType:$operand);
  let results = (outs Tla_ScalarValueType:$result);
  let assemblyFormat = [{
    $operand attr-dict `:` type($result)
  }];
}

def Tla_SimtSqrtOp  : Tla_SimtUnaryOp<"simt_sqrt",  "Per-thread scalar square root",
    "``math.sqrt``">;
def Tla_SimtExpOp   : Tla_SimtUnaryOp<"simt_exp",   "Per-thread scalar exponential",
    "``math.exp``">;
def Tla_SimtAbsOp   : Tla_SimtUnaryOp<"simt_abs",   "Per-thread scalar absolute value",
    "``math.absf``">;
// log is the one that cannot go through `math`: convert-hivm-to-std rewrites
// math.log into a call to the vln_1d_float vector template, which exists in
// meta_op.aiv.c220.bc but not in the c310 bitcode we link against, so the
// kernel fails at link with "undefined symbol: _mlir_ciface_vln_1d_float".
// Emitting the LLVM intrinsic directly skips that expansion and is selected
// inline, leaving no undefined symbol behind.
def Tla_SimtLogOp   : Tla_SimtUnaryOp<"simt_log",   "Per-thread scalar natural logarithm",
    "``llvm.intr.log``">;

def Tla_SimtLoadOp : Tla_Op<"simt_load", [MemoryEffects<[MemRead]>]> {
  let summary = "Per-thread tensor element read";
  let description = [{
    Read one element of a ``!tla.tensor`` from inside a ``tla.vec.func`` with
    ``mode = "simt"``: one thread, one element. The SIMT counterpart of
    ``tla.scalar_load``, which stays the general descriptor-based access.
    ``tla-vector-region-simt`` lowers this to ``memref.load`` on the outlined vector
    function's parameter, so no descriptor view is materialized.
  }];
  let arguments = (ins Tla_TensorType:$source, Variadic<Index>:$indices);
  let results = (outs Tla_ScalarValueType:$result);
  let assemblyFormat = [{
    $source `[` $indices `]` attr-dict `:` type($source) `->` type($result)
  }];
}

def Tla_SimtStoreOp : Tla_Op<"simt_store", [MemoryEffects<[MemWrite]>]> {
  let summary = "Per-thread tensor element write";
  let description = [{
    Write one element of a ``!tla.tensor`` from inside a ``tla.vec.func`` with
    ``mode = "simt"``. The SIMT counterpart of ``tla.scalar_store``; lowered by
    ``tla-vector-region-simt`` to ``memref.store`` on the outlined vector function's
    parameter.
  }];
  let arguments = (ins Tla_TensorType:$dest, Variadic<Index>:$indices,
                       Tla_ScalarValueType:$value);
  let assemblyFormat = [{
    $dest `[` $indices `]` `,` $value attr-dict `:` type($dest) `,` type($value)
  }];
}

def Tla_SimtAtomicAddOp : Tla_Op<"simt_atomic_add", [MemoryEffects<[MemRead, MemWrite]>]> {
  let summary = "Per-thread atomic pointer add";
  let description = [{
    Atomically add one per-thread scalar to the element a ``!tla.ptr`` points
    at, from inside a ``tla.vec.func`` with ``mode = "simt"``. Any number of
    threads may target the same address; the read-modify-write is atomic, so no
    update is lost. Typical uses are counters, histogram bins and reduction
    accumulators.

    The pointer carries the element type and address space, so this operation
    has no coordinate: the caller computes the element address, exactly as
    ``cute.arch.atomic_add(ptr, val)`` does. The pointee may be ``i32``/``ui32``/
    ``f32``/``f16``/``bf16`` in GM or UB; ``i64``/``ui64`` only in GM, which is
    what the hardware provides. The element is updated in place and the
    operation returns the value it replaced: the fetching form of the atom
    instruction. For ``f16``/``bf16`` the hardware return value is not the
    original element, so the frontend API returns nothing for those types.

    The pointer chain must be built inside the region -- from ``tla.tensor_ptr``
    (with ``tla.ptr_add`` for the element offset) or ``tla.inttoptr``. A
    ``!tla.ptr`` value cannot cross the region boundary, so ``tla-lower-ptr``
    folds the chain to its i64 address and ``tla-vector-region-simt`` rebuilds
    the pointer inside before lowering it to the
    ``llvm.hivm.atom.ADD.{G,S}.<type>`` intrinsic.
  }];
  let arguments = (ins Tla_PtrType:$ptr, Tla_ScalarValueType:$value);
  let results = (outs Tla_ScalarValueType:$result);
  let hasVerifier = 1;
  let assemblyFormat = [{
    $ptr `,` $value attr-dict `:` qualified(type($ptr)) `,` type($value) `->` type($result)
  }];
}
