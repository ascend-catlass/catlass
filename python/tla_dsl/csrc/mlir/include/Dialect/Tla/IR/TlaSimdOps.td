//===----------------------------------------------------------------------===//
// SIMD-mode vector-unit (AVE) ops: masks, register init, arithmetic, casts and
// UB <-> register movement. The per-thread counterparts live in TlaSimtOps.td.
//
// The regions these run in -- tla.vector and tla.vec.func -- are in
// TlaRegionOps.td.
//
// Included by Dialect/Tla/IR/Tla.td, the file mlir-tblgen is run on.
//===----------------------------------------------------------------------===//

def Tla_CreateMaskOp : Tla_Op<"create_mask", [Pure]> {
  let summary = "Create a vector mask from a fixed pattern";
  let description = [{
    Create an element-wise mask from a fixed AVE ``pge`` pattern, e.g.
    ``ALL``, ``ALLF``, ``VL8`` (lowest 8 lanes), ``H`` (lowest half), ``Q``
    (lowest quarter), ``M3``/``M4`` (multiples of 3/4).

    ``dtype`` fixes the mask lane count (256 bytes / dtype size) and must match
    the enclosing vector region's element width.
  }];
  let arguments = (ins StrAttr:$pattern, TypeAttr:$dtype);
  let results = (outs Tla_MaskSSAType:$result);
  let assemblyFormat = [{
    attr-dict `:` qualified(type($result))
  }];
  let hasVerifier = 1;
}

def Tla_UpdateMaskOp : Tla_Op<"update_mask", [Pure]> {
  let summary = "Tail mask plus the remaining element count";
  let description = [{
    Tail-loop helper mapping to AVE ``plt``. Given the number of elements still
    to process (``true_shape``), produce a mask (lane ``i`` is active
    iff ``i < true_shape``, saturating to all-true once ``true_shape`` reaches
    the lane count) together with the remaining count after this iteration,
    ``new_true_shape = true_shape - lanes`` where ``lanes = 256 bytes / dtype
    size`` (= 64 for f32). Thread ``new_true_shape`` back through the loop's
    carried values to mask successive chunks, including the partial tail.

    ``dtype`` fixes the lane count and therefore the decrement; it must match the
    enclosing vector region's element width.
  }];
  let arguments = (ins Index:$true_shape, TypeAttr:$dtype);
  let results = (outs Tla_MaskSSAType:$mask, Index:$new_true_shape);
  let assemblyFormat = [{
    $true_shape `,` $dtype attr-dict `:` qualified(type($mask)) `,` type($new_true_shape)
  }];
  let hasVerifier = 1;
}

// Register initializers: neither takes a tensor -- they fill the vector
// helper register that every op below operates on.

def Tla_FullOp : Tla_Op<"full", [Pure]> {
  let summary = "Fill vector lanes with a scalar or one-lane fragment";
  let description = [{
    Broadcast a value into the vector helper register type selected by the
    enclosing vector lowering context. ``value`` is either a plain scalar
    (lowered via AVE ``broadcast``, a mask-capable scalar splat) or a one-lane
    vector fragment (a ``tla.reduce`` result, lowered via AVE
    ``vector_broadcast``, which broadcasts lane 0 across the register). The
    latter avoids materializing a scalar register, which the SIMD backend cannot
    read from a vector lane.

    ``i1`` is not supported: the AVE ``broadcast`` op does not accept an ``i1``
    source, so boolean fills must be built another way.

    An optional ``mask`` predicates the broadcast: masked-out lanes are not
    written. It is passed through to the AVE broadcast op's lane mask; when
    absent an all-true mask is used.
  }];
  let arguments = (ins AnyTypeOf<[I8, I16, I32, I64, BF16, F16, F32,
                                  Tla_VectorSSAType]>:$value,
                       Optional<Tla_MaskSSAType>:$mask);
  let results = (outs Tla_VectorSSAType:$result);
  let assemblyFormat = [{
    $value (`mask` $mask^)? attr-dict `:` type($value)
    (`mask` qualified(type($mask))^)? `->` qualified(type($result))
  }];
  let hasVerifier = 1;
}

def Tla_ArangeOp : Tla_Op<"arange", [Pure]> {
  let summary = "Create monotonic lane indices in a vector register";
  let description = [{
    Fill the vector helper register with a monotonic sequence.

    With ``order = increase``(default), lane ``i`` holds ``base + i`` (i.e. ``start``, ``start + 1``, ...).
    With ``order = decrease``, lane ``i`` holds ``base + VL - 1 - i`` (i.e. ``start + VL - 1``, ``start + VL - 2``, ..., ``start``).

    Lowers to AVE ``vci`` with ``INCREASE`` / ``DECREASE`` mode.
  }];
  let arguments = (ins AnyTypeOf<[I8, I16, I32, I64]>:$start,
                       StrAttr:$order);
  let results = (outs Tla_VectorSSAType:$result);
  let assemblyFormat = [{
    $start attr-dict `:` type($start) `->` qualified(type($result))
  }];
  let hasVerifier = 1;
}

class Tla_BitwiseUnaryOp<string mnemonic, string opSummary, string opDescription>
    : Tla_Op<mnemonic, [Pure]> {
  let summary = opSummary;
  let description = opDescription # [{

    The ``mask`` operand predicates which lanes are computed; masked-out lanes
    are undefined/zeroed. When omitted, all lanes are active.
  }];
  let arguments = (ins AnyTypeOf<[Tla_VectorSSAType, Tla_MaskSSAType]>:$operand,
                       Optional<Tla_MaskSSAType>:$mask);
  let results = (outs AnyTypeOf<[Tla_VectorSSAType, Tla_MaskSSAType]>:$result);
  let assemblyFormat = [{
    $operand (`mask` $mask^)? attr-dict `:`
    qualified(type($operand)) (`mask` qualified(type($mask))^)?
    `->` qualified(type($result))
  }];
  let hasVerifier = 1;
}

class Tla_BitwiseBinaryOp<string mnemonic, string opSummary, string opDescription>
    : Tla_Op<mnemonic, [Pure]> {
  let summary = opSummary;
  let description = opDescription # [{

    The ``mask`` operand predicates which lanes are computed; masked-out lanes
    are undefined/zeroed. When omitted, all lanes are active.
  }];
  let arguments = (ins AnyTypeOf<[Tla_VectorSSAType, Tla_MaskSSAType]>:$lhs,
                       AnyTypeOf<[Tla_VectorSSAType, Tla_MaskSSAType]>:$rhs,
                       Optional<Tla_MaskSSAType>:$mask);
  let results = (outs AnyTypeOf<[Tla_VectorSSAType, Tla_MaskSSAType]>:$result);
  let assemblyFormat = [{
    $lhs `,` $rhs (`mask` $mask^)? attr-dict `:`
    qualified(type($lhs)) `,` qualified(type($rhs))
    (`mask` qualified(type($mask))^)?
    `->` qualified(type($result))
  }];
  let hasVerifier = 1;
}

def Tla_BitwiseNotOp : Tla_BitwiseUnaryOp<"bitwise_not",
    "Element-wise bitwise NOT",
    "Element-wise bitwise NOT on MaskSSA or VectorSSA values.">;

def Tla_BitwiseAndOp : Tla_BitwiseBinaryOp<"bitwise_and",
    "Element-wise bitwise AND",
    "Element-wise bitwise AND on MaskSSA or VectorSSA values.">;

def Tla_BitwiseOrOp : Tla_BitwiseBinaryOp<"bitwise_or",
    "Element-wise bitwise OR",
    "Element-wise bitwise OR on MaskSSA or VectorSSA values.">;

def Tla_BitwiseXorOp : Tla_BitwiseBinaryOp<"bitwise_xor",
    "Element-wise bitwise XOR",
    "Element-wise bitwise XOR on MaskSSA or VectorSSA values.">;

def Tla_CmpOp : Tla_Op<"cmp", [Pure]> {
  let summary = "Create a mask by comparing a vector with a vector or scalar";
  let description = [{
    Data-dependent mask: lane ``i`` is true when ``lhs[i] <mode> rhs[i]``.
    ``rhs`` may be a VectorSSA value or scalar SSA value. ``mode`` is one of
    ``lt``, ``le``, ``gt``, ``ge``, ``eq``, ``ne``. An optional mask controls
    which lanes participate in the compare.
  }];
  let arguments = (ins Tla_VectorSSAType:$lhs,
                       AnyTypeOf<[Tla_VectorSSAType, Tla_ScalarValueType]>:$rhs,
                       StrAttr:$mode,
                       Optional<Tla_MaskSSAType>:$mask);
  let results = (outs Tla_MaskSSAType:$result);
  let hasVerifier = 1;
  let assemblyFormat = [{
    $mode $lhs `,` $rhs (`mask` $mask^)? attr-dict `:`
    qualified(type($lhs)) `,` qualified(type($rhs)) (`mask` qualified(type($mask))^)? `->` qualified(type($result))
  }];
}

class Tla_VectorBinaryOp<string mnemonic, string opSummary, string opDescription>
    : Tla_Op<mnemonic, [Pure]> {
  let summary = opSummary;
  let description = opDescription # [{

    An optional mask controls which lanes are computed; masked-out lanes are
    undefined/zeroed.
  }];
  let arguments = (ins Tla_VectorSSAType:$lhs, Tla_VectorSSAType:$rhs,
                       Optional<Tla_MaskSSAType>:$mask);
  let results = (outs Tla_VectorSSAType:$result);
  let assemblyFormat = [{
    $lhs `,` $rhs (`mask` $mask^)? attr-dict `:`
    qualified(type($lhs)) `,` qualified(type($rhs)) (`mask` qualified(type($mask))^)? `->` qualified(type($result))
  }];
  let hasVerifier = 1;
}

class Tla_VectorShiftOp<string mnemonic, string opSummary, string opDescription>
    : Tla_Op<mnemonic, [Pure]> {
  let summary = opSummary;
  let description = opDescription # [{

    The signed rhs supplies a non-negative shift amount for each lane. The
    required mask controls which lanes are computed; masked-out lanes are
    zeroed.
  }];
  let arguments = (ins Tla_VectorSSAType:$lhs, Tla_VectorSSAType:$rhs,
                       Tla_MaskSSAType:$mask);
  let results = (outs Tla_VectorSSAType:$result);
  let assemblyFormat = [{
    $lhs `,` $rhs `mask` $mask attr-dict `:`
    qualified(type($lhs)) `,` qualified(type($rhs)) `mask`
    qualified(type($mask)) `->` qualified(type($result))
  }];
  let hasVerifier = 1;
}

def Tla_AddOp : Tla_VectorBinaryOp<"add",
    "Vector add on VectorSSA values",
    "Element-wise add on VectorSSA values.">;

def Tla_SubOp : Tla_VectorBinaryOp<"sub",
    "Vector subtract on VectorSSA values",
    "Element-wise subtract on VectorSSA values.">;

def Tla_MulOp : Tla_VectorBinaryOp<"mul",
    "Vector multiply on VectorSSA values",
    "Element-wise multiply on VectorSSA values.">;

def Tla_DivOp : Tla_VectorBinaryOp<"div",
    "Vector divide on VectorSSA values",
    "Element-wise divide on VectorSSA values.">;

def Tla_MaxOp : Tla_VectorBinaryOp<"max",
    "Vector maximum on VectorSSA values",
    "Element-wise maximum on VectorSSA values.">;

def Tla_MinOp : Tla_VectorBinaryOp<"min",
    "Vector minimum on VectorSSA values",
    "Element-wise minimum on VectorSSA values.">;

def Tla_ShiftLeftOp : Tla_VectorShiftOp<"shift_left",
    "Vector-vector left shift on VectorSSA values",
    "Element-wise left shift by the signed amounts in the rhs VectorSSA value.">;

def Tla_ShiftRightOp : Tla_VectorShiftOp<"shift_right",
    "Vector-vector right shift on VectorSSA values",
    "Element-wise right shift by the signed amounts in the rhs VectorSSA value.">;

def Tla_GatherOp : Tla_Op<"gather", [Pure]> {
  let summary = "Vector gather from UB tile or register";
  let description = [{
    Gathers elements from ``x`` into ``result`` according to the indices in ``y``.

    Two source forms are supported:

    * **UB→Reg** (``x`` is a UB ``!tla.tensor``): per-lane indexed load.
      ``y`` must be ``i32``. Optional ``mask`` predicates lanes.
    * **Reg→Reg** (``x`` is ``!tla.vector``): register permute
      ``result[i] = x[y[i] % VL]``. ``y`` must be an integer with the same
      element bit-width as ``x`` (``i8``/``u8``, ``i16``/``u16``, or
      ``i32``/``u32``). No ``mask`` in this form.
  }];
  let arguments = (ins
    AnyTypeOf<[Tla_TensorType, Tla_VectorSSAType]>:$x,
    Tla_VectorSSAType:$y,
    Optional<Tla_MaskSSAType>:$mask
  );
  let results = (outs Tla_VectorSSAType:$result);
  let assemblyFormat = [{
    $x `,` $y (`mask` $mask^)? attr-dict `:`
    type($x) `,` qualified(type($y)) (`mask` qualified(type($mask))^)? `->` qualified(type($result))
  }];
  let hasVerifier = 1;
}

def Tla_HistogramsOp : Tla_Op<"histograms", [Pure]> {
  let summary = "Reg FREQUENCY/ACCUMULATE histogram via AIV bc";
  let description = [{
  Compute a register-resident histogram half (128 u16 lanes):

  * ``type = frequency`` → CCE ``dhistv2`` (per-value counts in a half-range):
    ``mode = bin0`` → ``dst[v] += #{src == v}`` for ``v ∈ [0, 127]``;
    ``mode = bin1`` → ``dst[v - 128] += #{src == v}`` for ``v ∈ [128, 255]``.
  * ``type = accumulate`` → CCE ``chistv2`` (CDF over full u8, then split):
    ``mode = bin0`` → ``dst[n] += #{src <= n}`` for ``n ∈ [0, 127]``;
    ``mode = bin1`` → ``dst[n] += #{src <= 128 + n}`` for ``n ∈ [0, 127]``
    (low-half inputs contribute; not a ``[128, 255]``-only bucket).

  ``src`` must be an 8-bit integer VectorSSA (256 lanes / VL). ``dst`` is the
  **required** u16 accumulator (128 lanes / VL); the op accumulates onto the
  prior ``dst`` value. Callers zero explicitly (e.g. ``tla.full(0, ...)``)
  for a fresh histogram. Optional ``mask`` predicates which ``src`` lanes
  participate. ``result`` has the same type as ``dst``.

  Lowering is **AIV bc only** (``func.call @histograms_*``), not HIVMAVE.
  }];
  let arguments = (ins
    Tla_VectorSSAType:$src,
    Tla_VectorSSAType:$dst,
    Optional<Tla_MaskSSAType>:$mask,
    // histType avoids clashing with Operation::getType.
    Tla_HistogramsBinAttr:$mode,
    Tla_HistogramsTypeAttr:$histType
  );
  let results = (outs Tla_VectorSSAType:$result);
  let assemblyFormat = [{
    $src `,` $dst (`mask` $mask^)?
    `mode` $mode
    `type` $histType
    attr-dict `:`
    qualified(type($src)) `,` qualified(type($dst))
    (`mask` qualified(type($mask))^)?
    `->` qualified(type($result))
  }];
  let hasVerifier = 1;
}

def Tla_SqueezeOp : Tla_Op<"squeeze", [Pure]> {
  let summary = "Mask-compress vector lanes into low indices";
  let description = [{
  Copy lanes of ``src`` selected by ``mask`` into ``result`` starting at lane 0.
  Remaining lanes in ``result`` are zeroed. This matches AscendC ``Squeeze``
  semantics (valid elements packed from low to high, not value-sorted).
  }];
  let arguments = (ins Tla_VectorSSAType:$src, Tla_MaskSSAType:$mask);
  let results = (outs Tla_VectorSSAType:$result);
  let assemblyFormat = [{
    $src `,` $mask attr-dict `:`
    qualified(type($src)) `,` qualified(type($mask)) `->` qualified(type($result))
  }];
  let hasVerifier = 1;
}

def Tla_CastOp : Tla_Op<"cast", [Pure]> {
  let summary = "Vector element-type conversion on VectorSSA values";
  let description = [{
    Convert a register-resident VectorSSA value from one element type to another
    (e.g. ``f32`` -> ``f16``, ``f32`` -> ``i32``). The ``trait`` attribute carries
    the cast knobs (rounding, saturation, register slot). An optional mask
    predicates which lanes are converted.
  }];
  let arguments = (ins Tla_VectorSSAType:$source, DenseI32ArrayAttr:$trait,
                       Optional<Tla_MaskSSAType>:$mask);
  let results = (outs Tla_VectorSSAType:$result);
  let assemblyFormat = [{
    $source `,` $trait (`mask` $mask^)? attr-dict `:`
    qualified(type($source)) (`mask` qualified(type($mask))^)? `->` qualified(type($result))
  }];
  let hasVerifier = 1;
}

def Tla_BitCastOp : Tla_Op<"bitcast", [Pure]> {
  let summary = "Same-width vector bit reinterpretation on VectorSSA values";
  let description = [{
    Reinterpret the bits of a register-resident VectorSSA value as another
    element type of the **same bit width** (e.g. ``f32`` -> ``i32``,
    ``f16`` -> ``i16``). No numeric conversion takes place and no hardware
    instruction is emitted: the op lowers to ``vector.bitcast`` and costs zero
    cycles.

    The lane count is preserved, so source and result must have the same
    ``valid_lanes`` and the same element width. Changing the lane count is
    ``tla.cast``'s job, not this op's. Because nothing is computed, there is no
    mask and no cast trait.
  }];
  let arguments = (ins Tla_VectorSSAType:$source);
  let results = (outs Tla_VectorSSAType:$result);
  let assemblyFormat = [{
    $source attr-dict `:` qualified(type($source)) `->` qualified(type($result))
  }];
  let hasVerifier = 1;
}

def Tla_WhereOp : Tla_Op<"where", [Pure]> {
  let summary = "Select on VectorSSA or MaskSSA values";
  let description = [{
    Select between ``x`` and ``y`` under ``mask``. For each logical lane,
    an active bit in ``mask`` selects ``x`` and an inactive bit selects ``y``.
    ``x``, ``y``, and ``result`` must all be VectorSSA values or all be
    MaskSSA values. The MaskSSA form additionally requires all four values to
    have the same ``!tla.mask<N>`` type.
  }];
  let arguments = (ins
      Tla_MaskSSAType:$mask,
      AnyTypeOf<[Tla_VectorSSAType, Tla_MaskSSAType]>:$x,
      AnyTypeOf<[Tla_VectorSSAType, Tla_MaskSSAType]>:$y);
  let results = (outs
      AnyTypeOf<[Tla_VectorSSAType, Tla_MaskSSAType]>:$result);
  let assemblyFormat = [{
    $mask `,` $x `,` $y attr-dict `:`
    qualified(type($mask)) `,` qualified(type($x)) `,` qualified(type($y)) `->` qualified(type($result))
  }];
  let hasVerifier = 1;
}

class Tla_VectorScalarBinaryOp<string mnemonic, string opSummary, string opDescription>
    : Tla_Op<mnemonic, [Pure]> {
  let summary = opSummary;
  let description = opDescription # [{

    The right-hand side is a scalar SSA value. An optional mask controls which
    lanes are computed; masked-out lanes are undefined/zeroed.
  }];
  let arguments = (ins Tla_VectorSSAType:$lhs, Tla_ScalarValueType:$rhs,
                       Optional<Tla_MaskSSAType>:$mask);
  let results = (outs Tla_VectorSSAType:$result);
  let assemblyFormat = [{
    $lhs `,` $rhs (`mask` $mask^)? attr-dict `:`
    qualified(type($lhs)) `,` type($rhs) (`mask` qualified(type($mask))^)? `->` qualified(type($result))
  }];
  let hasVerifier = 1;
}

class Tla_VectorScalarShiftOp<string mnemonic, string opSummary, string opDescription>
    : Tla_Op<mnemonic, [Pure]> {
  let summary = opSummary;
  let description = opDescription # [{

    The rhs is a signed i16 scalar shift amount. The required mask controls
    which lanes are computed; masked-out lanes are zeroed.
  }];
  let arguments = (ins Tla_VectorSSAType:$lhs, I16:$rhs,
                       Tla_MaskSSAType:$mask);
  let results = (outs Tla_VectorSSAType:$result);
  let assemblyFormat = [{
    $lhs `,` $rhs `mask` $mask attr-dict `:`
    qualified(type($lhs)) `,` type($rhs) `mask`
    qualified(type($mask)) `->` qualified(type($result))
  }];
  let hasVerifier = 1;
}

def Tla_AddsOp : Tla_VectorScalarBinaryOp<"adds",
    "Vector-scalar add on VectorSSA values",
    "Element-wise add of a scalar to VectorSSA values.">;

def Tla_SubsOp : Tla_VectorScalarBinaryOp<"subs",
    "Vector-scalar subtract on VectorSSA values",
    "Element-wise subtract of a scalar from VectorSSA values.">;

def Tla_MulsOp : Tla_VectorScalarBinaryOp<"muls",
    "Vector-scalar multiply on VectorSSA values",
    "Element-wise multiply of VectorSSA values by a scalar.">;

def Tla_MaxsOp : Tla_VectorScalarBinaryOp<"maxs",
    "Vector-scalar maximum on VectorSSA values",
    "Element-wise maximum of VectorSSA values and a scalar.">;

def Tla_MinsOp : Tla_VectorScalarBinaryOp<"mins",
    "Vector-scalar minimum on VectorSSA values",
    "Element-wise minimum of VectorSSA values and a scalar.">;

def Tla_DivsOp : Tla_VectorScalarBinaryOp<"divs",
    "Vector-scalar divide on VectorSSA values",
    "Element-wise divide of VectorSSA values by a scalar.">;

def Tla_ShiftLeftsOp : Tla_VectorScalarShiftOp<"shift_lefts",
    "Vector-scalar left shift on VectorSSA values",
    "Element-wise left shift by a signed i16 scalar amount.">;

def Tla_ShiftRightsOp : Tla_VectorScalarShiftOp<"shift_rights",
    "Vector-scalar right shift on VectorSSA values",
    "Element-wise right shift by a signed i16 scalar amount.">;

def Tla_ReduceOp : Tla_Op<"reduce", [Pure]> {
  let summary = "Vector reduction on VectorSSA values";
  let description = [{
    Reduce a register-resident VectorSSA value to a one-lane VectorSSA value.
  }];
  let arguments = (ins Tla_VectorSSAType:$operand, Optional<Tla_MaskSSAType>:$mask,
                       StrAttr:$kind);
  let results = (outs Tla_VectorSSAType:$result);
  let assemblyFormat = [{
    $operand (`mask` $mask^)? attr-dict `:`
    qualified(type($operand)) (`mask` qualified(type($mask))^)? `->` qualified(type($result))
  }];
  let hasVerifier = 1;
}

def Tla_ReduceIntraBlockOp : Tla_Op<"reduce_intra_block", [Pure]> {
  let summary = "Per-32B-block reduction on VectorSSA values (packed result)";
  let description = [{
    Reduce each 32-byte data block of a register-resident VectorSSA value under
    a mask. Each block's result
    is written contiguously to the least significant lanes of the result vector;
    the remaining lanes of each block are 0. The result mirrors the operand
    type, except that `add` on i16/ui16 returns the widened 32-bit instruction
    destination (i32/ui32, half the lanes), with no cast back to 16 bits.
  }];
  let arguments = (ins Tla_VectorSSAType:$operand, Tla_MaskSSAType:$mask,
                       StrAttr:$kind);
  let results = (outs Tla_VectorSSAType:$result);
  let assemblyFormat = [{
    $operand `mask` $mask attr-dict `:`
    qualified(type($operand)) `mask` qualified(type($mask)) `->` qualified(type($result))
  }];
  let hasVerifier = 1;
}

def Tla_InterleaveOp : Tla_Op<"interleave", [Pure]> {
  let summary = "Interleave on VectorSSA or MaskSSA values";
  let arguments = (ins AnyTypeOf<[Tla_VectorSSAType, Tla_MaskSSAType]>:$src0,
                       AnyTypeOf<[Tla_VectorSSAType, Tla_MaskSSAType]>:$src1);
  let results = (outs AnyTypeOf<[Tla_VectorSSAType, Tla_MaskSSAType]>:$dst0,
                      AnyTypeOf<[Tla_VectorSSAType, Tla_MaskSSAType]>:$dst1);
  let assemblyFormat = [{
    $src0 `,` $src1 attr-dict `:`
    qualified(type($src0)) `,` qualified(type($src1)) `->` qualified(type($dst0)) `,` qualified(type($dst1))
  }];
  let hasVerifier = 1;
}

def Tla_DeInterleaveOp : Tla_Op<"deinterleave", [Pure]> {
  let summary = "Deinterleave on VectorSSA or MaskSSA values";
  let arguments = (ins AnyTypeOf<[Tla_VectorSSAType, Tla_MaskSSAType]>:$src0,
                       AnyTypeOf<[Tla_VectorSSAType, Tla_MaskSSAType]>:$src1);
  let results = (outs AnyTypeOf<[Tla_VectorSSAType, Tla_MaskSSAType]>:$dst0,
                      AnyTypeOf<[Tla_VectorSSAType, Tla_MaskSSAType]>:$dst1);
  let assemblyFormat = [{
    $src0 `,` $src1 attr-dict `:`
    qualified(type($src0)) `,` qualified(type($src1)) `->` qualified(type($dst0)) `,` qualified(type($dst1))
  }];
  let hasVerifier = 1;
}

class Tla_VectorUnaryOp<string mnemonic, string opSummary, string opDescription>
    : Tla_Op<mnemonic, [Pure]> {
  let summary = opSummary;
  let description = opDescription # [{

    An optional mask controls which lanes are computed; masked-out lanes are
    undefined/zeroed.
  }];
  let arguments = (ins Tla_VectorSSAType:$operand,
                       Optional<Tla_MaskSSAType>:$mask);
  let results = (outs Tla_VectorSSAType:$result);
  let assemblyFormat = [{
    $operand (`mask` $mask^)? attr-dict `:`
    qualified(type($operand)) (`mask` qualified(type($mask))^)?
    `->` qualified(type($result))
  }];
  let hasVerifier = 1;
}

def Tla_ExpOp : Tla_VectorUnaryOp<"exp",
    "Vector exponential on VectorSSA values",
    "Element-wise exponential on VectorSSA values.">;

def Tla_LogOp : Tla_VectorUnaryOp<"log",
    "Vector natural logarithm on VectorSSA values",
    "Element-wise natural logarithm on VectorSSA values.">;

def Tla_SqrtOp : Tla_VectorUnaryOp<"sqrt",
    "Vector square root on VectorSSA values",
    "Element-wise square root on VectorSSA values.">;

def Tla_AbsOp : Tla_VectorUnaryOp<"abs",
    "Vector absolute value on VectorSSA values",
    "Element-wise absolute value on VectorSSA values.">;

def Tla_NegOp : Tla_VectorUnaryOp<"neg",
    "Vector negate on VectorSSA values",
    "Element-wise negate on VectorSSA values.">;

// UB <-> vector/mask register. Both require a ``tla.vec.func`` region,
// which is why they live here and not in TlaMemoryOps.td.

def Tla_LoadOp : Tla_Op<"load", []> {
  let summary = "Load";
  let description = [{
    Load a UB tensor tile into vector or mask SSA. Dispatch is by result type:

    * ``!tla.vector`` — data load. Optional ``load_dist`` / ``unaligned_ub_access``.
      Dual-destination ``dintlv_b32`` produces a second result (even/odd b32
      elements), requiring aligned access. Unaligned path
      lowers to ``ave.hir.vload`` with ``vldas``/``vldus`` via HIVMAVE.
      Optional compile-time ``block_stride`` (i32 attribute) enables strided
      block load (AscendC ``vsldb``): one instruction gathers 8 DataBlocks
      whose heads are ``block_stride`` DataBlocks (32B units) apart;
      ``block_stride == 0`` repeats the first DataBlock. ``repeat_stride``
      (POST_MODE_NORMAL) is a compile-time address pre-offset in 32B
      DataBlocks. Strided load is mutually exclusive with ``load_dist`` and
      the dual result. PostMode UPDATE is ``tla.load_update``, not this op.

    * ``!tla.mask`` — MaskReg load (AscendC ``LoadAlign(MaskReg&)`` /
      ``MaskDist`` / ``plds``). Source UB element type must be a 1/2/4-byte
      scalar (int or float). ``N`` on ``!tla.mask<N>`` is
      ``256 / sizeof(elem)`` from the UB element type; the tile supplies the
      address. Optional ``load_dist`` ``norm``/``us``/``ds`` maps to
      ``hivm.plds.b8`` dist 0/1/2 (``DIST_NORM``/``US``/``DS``). ``DIST_NORM``
      lowers via ``ave.hir.vload <NORM>`` on an ``i1`` memref view;
      ``us``/``ds`` lower via AIV bc (NPU-IR i1→``plds`` hardcodes NORM).

    Must be nested inside a ``tla.vec.func`` region.
  }];
  let arguments = (ins Tla_TensorType:$source,
                       OptionalAttr<I32Attr>:$block_stride,
                       OptionalAttr<I32Attr>:$repeat_stride,
                       OptionalAttr<Tla_LoadDistAttr>:$load_dist,
                       OptionalAttr<UnitAttr>:$unaligned_ub_access);
  // Second result: dintlv even/odd vector only.
  let results = (outs AnyTypeOf<[Tla_VectorSSAType, Tla_MaskSSAType]>:$result,
                      Optional<Tla_VectorSSAType>:$result2);
  let assemblyFormat = [{
    $source attr-dict `:`
    type($source) `->` qualified(type($result)) (`,` qualified(type($result2))^)?
  }];
  let hasVerifier = 1;
}

def Tla_StoreOp : Tla_Op<"store", [AttrSizedOperandSegments]> {
  let summary = "Store";
  let description = [{
    Store vector or mask SSA into a UB tensor tile. Dispatch is by source type:

    * ``!tla.vector`` — data store. Optional predicate ``mask`` controls which
      lanes are written. Optional ``unaligned_ub_access`` lowers to unaligned
      UB store (``vstas``/``vstus`` via HIVMAVE). ``block_stride`` selects
      DATA_BLOCK_COPY BlockStore. PostMode UPDATE is ``tla.store_update``.
    * ``!tla.mask`` — continuous-aligned MaskReg store (AscendC
      ``StoreAlign(..., MaskReg&)`` / ``DIST_NORM`` / ``psts``). Destination
      UB element type must be a 1/2/4-byte scalar (int or float). Spill size
      is ``N/8`` from the ``MaskSSA`` source; the UB tile supplies the
      address. Predicate ``mask`` and ``unaligned_ub_access`` are not
      allowed. Lowers via ``ave.hir.masked_store <NORM_B8>`` on an ``i1``
      memref view to ``hivm.psts.b8``.

    Destination must be UB. Must be nested inside a ``tla.vec.func`` region.
  }];
  let arguments = (ins Tla_TensorType:$dest,
                       AnyTypeOf<[Tla_VectorSSAType, Tla_MaskSSAType]>:$source,
                       Optional<Tla_MaskSSAType>:$mask,
                       Optional<Tla_IntegerOrIndexType>:$block_stride,
                       OptionalAttr<Tla_StoreDistAttr>:$store_dist,
                       OptionalAttr<UnitAttr>:$unaligned_ub_access);
  let results = (outs);
  let assemblyFormat = [{
    $dest `,` $source (`mask` $mask^)? (`stride` $block_stride^)?
    attr-dict `:`
    type($dest) `,` qualified(type($source)) (`mask` qualified(type($mask))^)?
    (`block_stride` qualified(type($block_stride))^)?
  }];
  let hasVerifier = 1;
}

def Tla_PackOp : Tla_Op<"pack", [Pure]> {
  let summary = "Pack vector elements or predicates into a destination half";
  let description = [{
    For vectors, truncates integer elements to their low half of bits and
    places them in the selected destination half. For masks, compacts the
    source into the selected half of a wider predicate domain.
    The destination element type is determined by the result type.
  }];
  let arguments = (ins
      AnyTypeOf<[Tla_VectorSSAType, Tla_MaskSSAType]>:$src,
      Tla_RegPartAttr:$part);
  let results = (outs
      AnyTypeOf<[Tla_VectorSSAType, Tla_MaskSSAType]>:$result);
  let assemblyFormat = [{
    $src `,` $part attr-dict `:`
    qualified(type($src)) `->` qualified(type($result))
  }];
  let hasVerifier = 1;
}

def Tla_UnpackOp : Tla_Op<"unpack", [Pure]> {
  let summary = "Extend integer elements from the selected source half";
  let description = [{
    Reads the selected source half and widens integer elements. Signed
    integers sign-extend; unsigned integers zero-extend. Source and result
    must have matching signedness and the result element width is doubled.
  }];
  let arguments = (ins Tla_VectorSSAType:$src, Tla_RegPartAttr:$part);
  let results = (outs Tla_VectorSSAType:$result);
  let assemblyFormat = [{
    $src `,` $part attr-dict `:`
    qualified(type($src)) `->` qualified(type($result))
  }];
  let hasVerifier = 1;
}

def Tla_LoadUpdateOp : Tla_Op<"load_update", []> {
  let summary = "PostMode UPDATE load (aligned or DATA_BLOCK_COPY)";
  let description = [{
    UB load with AscendC ``POST_MODE_UPDATE``. Results are the loaded
    ``!tla.vector`` and the next-window ``!tla.ptr``.

    * Without ``block_stride``: continuous aligned load. ``post_update_stride``
      is an **element** count. VectorRegion emits HIVM ``vldsx1.post``.
    * With ``block_stride``: AscendC ``LoadAlign<T, DATA_BLOCK_COPY,
      POST_MODE_UPDATE>`` (``vsldb`` POST). Both ``block_stride`` and
      ``post_update_stride`` are 32B DataBlock counts. There is no HIVMC
      ``vsldb.post``, so VectorRegion lowers through an AIV bc that returns
      the hardware pointer.

    The frontend builds a coord-0 tile from the pointer and yields the tile.
    UnAlign UPDATE is not this op.
  }];
  let arguments = (ins Tla_TensorType:$source,
                       Tla_IntegerOrIndexType:$post_update_stride,
                       Optional<Tla_IntegerOrIndexType>:$block_stride);
  let results = (outs Tla_VectorSSAType:$result,
                      Tla_PtrType:$next_ptr);
  let assemblyFormat = [{
    $source `post_update_stride` $post_update_stride
    (`block_stride` $block_stride^)? attr-dict `:`
    type($source) `post_update_stride` qualified(type($post_update_stride))
    (`block_stride` qualified(type($block_stride))^)?
    `->` qualified(type($result)) `,` qualified(type($next_ptr))
  }];
  let hasVerifier = 1;
}

def Tla_StoreUpdateOp : Tla_Op<"store_update", [AttrSizedOperandSegments]> {
  let summary = "PostMode UPDATE store (aligned or DATA_BLOCK_COPY)";
  let description = [{
    UB store with AscendC ``POST_MODE_UPDATE``. Result is the next-window
    ``!tla.ptr``. Optional predicate ``mask``.

    * Without ``block_stride``: continuous aligned store. ``post_update_stride``
      is an **element** count. VectorRegion emits HIVM ``vstsx1.post``.
    * With ``block_stride``: AscendC ``StoreAlign<T, DATA_BLOCK_COPY,
      POST_MODE_UPDATE>`` (``vsstb`` POST). Both strides are 32B DataBlock
      counts. There is no HIVMC ``vsstb.post``, so VectorRegion lowers through
      an AIV bc that returns the hardware pointer.

    The frontend builds a coord-0 tile from the pointer and yields the tile.
    UnAlign UPDATE is not this op.
  }];
  let arguments = (ins Tla_TensorType:$dest,
                       Tla_VectorSSAType:$source,
                       Tla_IntegerOrIndexType:$post_update_stride,
                       Optional<Tla_MaskSSAType>:$mask,
                       Optional<Tla_IntegerOrIndexType>:$block_stride);
  let results = (outs Tla_PtrType:$next_ptr);
  let assemblyFormat = [{
    $dest `,` $source `post_update_stride` $post_update_stride
    (`mask` $mask^)? (`block_stride` $block_stride^)?
    attr-dict `:`
    type($dest) `,` qualified(type($source))
    `post_update_stride` qualified(type($post_update_stride))
    (`mask` qualified(type($mask))^)?
    (`block_stride` qualified(type($block_stride))^)?
    `->` qualified(type($next_ptr))
  }];
  let hasVerifier = 1;
}

// UnAlign UPDATE (dense tile_extent): each op is self-contained dual-instr
// (load: vldas + vldus.post; store: empty ureg + vstus.post + vstas). No
// cross-trip ureg SSA / stream Pre-Empty. HIVM ``$cceVL``/``$eleSize`` =
// ``post_update_stride×sizeof(T)``; TLA ``next_ptr`` is the HW ``newPtr``.
def Tla_LoadUnalignUpdateOp : Tla_Op<"load_unalign_update", []> {
  let summary = "LoadUnAlign POST_MODE_UPDATE (vldas + vldus.post; HW next_ptr)";
  let description = [{
    Loads one unaligned window. ``post_update_stride`` is an element count
    (AscendC ``postUpdateStride``; frontend fills it from tile ``origin_shape``).
    Lowering passes ``stride×sizeof(T)`` as HIVM ``$cceVL`` and returns the HW
    ``newPtr``. Each op emits a local ``vldas`` (no carried ureg).
  }];
  let arguments = (ins Tla_TensorType:$source,
                       Tla_IntegerOrIndexType:$post_update_stride);
  let results = (outs Tla_VectorSSAType:$result,
                      Tla_PtrType:$next_ptr);
  let assemblyFormat = [{
    $source `post_update_stride` $post_update_stride attr-dict `:`
    type($source)
    `post_update_stride` qualified(type($post_update_stride))
    `->` qualified(type($result)) `,` qualified(type($next_ptr))
  }];
  let hasVerifier = 1;
}

def Tla_StoreUnalignUpdateOp : Tla_Op<"store_unalign_update", []> {
  let summary = "StoreUnAlign POST_MODE_UPDATE (empty ureg + vstus.post + vstas; HW next_ptr)";
  let description = [{
    Stores one unaligned window. ``post_update_stride`` is an element count
    (AscendC ``postUpdateStride``; frontend fills it from tile ``origin_shape``).
    Same HIVM ``$eleSize`` / HW ``newPtr`` rule as ``load_unalign_update``.
    Each op emits empty ureg + ``vstus.post`` + ``vstas`` (no carried ureg).
  }];
  let arguments = (ins Tla_TensorType:$dest,
                       Tla_VectorSSAType:$source,
                       Tla_IntegerOrIndexType:$post_update_stride);
  let results = (outs Tla_PtrType:$next_ptr);
  let assemblyFormat = [{
    $dest `,` $source `post_update_stride` $post_update_stride attr-dict `:`
    type($dest) `,` qualified(type($source))
    `post_update_stride` qualified(type($post_update_stride))
    `->` qualified(type($next_ptr))
  }];
  let hasVerifier = 1;
}
