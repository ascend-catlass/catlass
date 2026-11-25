//===----------------------------------------------------------------------===//
// Scalar-unit ops: the ``scalar_*`` family, which runs on the core's own
// scalar unit rather than on the vector or cube units.
//
// Like the arch.* queries these are not tied to one execution mode -- they are
// valid in a tla.cube region, in a tla.vec.func in either mode, and in the
// plain kernel body.
//
// Lowered by TlaLowerScalarAccessPass (load/store) and TlaLowerScalarOpsPass
// (round cast, uint64 bit queries).
//
// Included by Dialect/Tla/IR/Tla.td, the file mlir-tblgen is run on.
//===----------------------------------------------------------------------===//

def Tla_ScalarLoadOp : Tla_Op<"scalar_load", [MemoryEffects<[MemRead]>]> {
  let summary = "Scalar tensor element read";
  let description = [{
    Read one scalar from a GM or UB ``!tla.tensor``. Indices are relative to the
    tensor view; its coordinate is applied while the descriptor is materialized
    as a memref. GM and UB support logical rank 1 or 2. UB accesses must be
    nested inside ``tla.vector`` and may also be nested inside ``tla.vec.func``.
  }];
  let arguments = (ins Tla_TensorType:$source, Variadic<Index>:$indices);
  let results = (outs AnyType:$result);
  let hasVerifier = 1;
  let assemblyFormat = [{
    $source `[` $indices `]` attr-dict `:` type($source) `->` type($result)
  }];
}

def Tla_ScalarStoreOp : Tla_Op<"scalar_store", [MemoryEffects<[MemWrite]>]> {
  let summary = "Scalar tensor element write";
  let description = [{
    Write one scalar to a GM or UB ``!tla.tensor``. Indices are relative to the
    tensor view; its coordinate is applied while the descriptor is materialized
    as a memref. GM and UB support logical rank 1 or 2. UB accesses must be
    nested inside ``tla.vector`` and may also be nested inside ``tla.vec.func``.
  }];
  let arguments = (ins Tla_TensorType:$dest, Variadic<Index>:$indices, AnyType:$value);
  let hasVerifier = 1;
  let assemblyFormat = [{
    $dest `[` $indices `]` `,` $value attr-dict `:` type($dest) `,` type($value)
  }];
}

def Tla_ScalarRoundCastOp : Tla_Op<"scalar_round_cast", [Pure]> {
  let summary = "Scalar f32 conversion with an explicit rounding mode";
  let description = [{
    Convert one ``f32`` scalar under a named rounding mode, on the core's own
    scalar unit. For an ``i32`` result ``mode`` is one of ``rn`` (nearest, ties
    to even), ``ra`` (nearest, ties away from zero), ``rd`` (floor) or ``ru``
    (ceil). For an ``f16`` result the only mode is ``o`` (round to odd), which
    is the only f32-to-f16 instruction the scalar unit has.

    This is NOT ``tla.simt_cast``'s ``fptosi``, which always truncates toward
    zero, nor ``tla.cast``, which converts a whole VectorSSA under the AVE cast
    knobs. Each mode here is a single scalar instruction --
    ``conv_f322s32{r,a,f,c}`` for the ``i32`` result, ``conv_f322f16o`` for the
    ``f16`` one -- and each is reachable only through the meta-op bitcode:
    ``math.floor``/``ceil``/``round``/``roundeven`` lower to a libm call that
    leaves an undefined symbol, and the ``llvm.hivm.CONV.*`` intrinsics are
    unknown to the LLVM hivmc links against. ``tla-lower-scalar-ops`` therefore
    rewrites this into a call to ``_mlir_ciface_tla_cast_f32_to_i32_<mode>`` or
    ``_mlir_ciface_tla_cast_f32_to_f16_o``.

    Truncation toward zero has no mode here because the hardware has no
    instruction for it -- the family is ``conv_f322s32{r,a,f,c}`` with no
    toward-zero form -- so the frontend emits ``arith.fptosi`` directly.

    ``f32`` to ``i32`` and ``f32`` to ``f16`` are the only conversions this op
    covers because they are the only ones the scalar unit has. Enumerating every
    ``conv_<src>2<dst><mode>`` name over the 11 scalar types and 7 rounding
    suffixes -- 770 candidates -- and compiling them with ccec for dav-c310
    leaves exactly five declared, on both AIC and AIV:
    ``conv_f322s32{r,a,f,c}`` and ``conv_f322f16o``. The larger ``conv_*``
    family in the CANN headers is the A2/A3 *vector* conversion set, not this
    one.

    ``f32`` to ``f16`` is not the default narrowing: ``arith.truncf`` already
    gives correct round-to-nearest-even there, verified on device against exact
    ties, whereas this instruction rounds to odd. It is reachable only when the
    caller names the mode.
  }];
  let arguments = (ins F32:$source, StrAttr:$mode);
  let results = (outs AnyTypeOf<[I32, F16]>:$result);
  let hasVerifier = 1;
  let assemblyFormat = [{
    $mode $source attr-dict `:` type($source) `to` type($result)
  }];
}

def Tla_ScalarBitOp : Tla_Op<"scalar_bit_op", [Pure]> {
  let summary = "Scalar uint64 bit-manipulation query";
  let description = [{
    Query the binary representation of a ``uint64`` scalar on the core's own
    scalar unit. ``mode`` names the query:

    - ``bcnt0`` / ``bcnt1``: number of ``0`` / ``1`` bits.
    - ``clz``: number of leading zeros before the most significant ``1``; ``64``
      when the input is ``0``.
    - ``sff0`` / ``sff1``: position of the first ``0`` / ``1`` bit starting from
      the least significant bit; ``-1`` when there is none.

    The operand and result are signless ``i64``: the frontend's ``UInt64`` is
    the ``uint64_t`` argument and the query counts/positions are returned as
    ``int64``. Each mode is a single scalar instruction -- ``bcnt0`` / ``bcnt1``,
    ``clz``, ``sff0`` / ``sff1`` -- reachable only through the meta-op bitcode.
    The vector ``hivm.vbcnt.x`` / ``hivm.vcls.*`` ops are 32-lane operations, so
    they cannot lower this 64-bit scalar query; ``tla-lower-scalar-ops``
    therefore rewrites this into a call to ``_mlir_ciface_bcnt0`` and friends.
  }];
  let arguments = (ins I64:$value, StrAttr:$mode);
  let results = (outs I64:$result);
  let hasVerifier = 1;
  let assemblyFormat = [{
    $mode $value attr-dict `:` type($value) `to` type($result)
  }];
}
