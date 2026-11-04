//===----------------------------------------------------------------------===//
// Enum cases, enums and enum attributes of the Tla dialect.
//
// Included by Dialect/Tla/IR/Tla.td, the file mlir-tblgen is run on.
//===----------------------------------------------------------------------===//

def Tla_LayoutTag_Unknown : I32EnumAttrCase<"Unknown", 0>;
def Tla_LayoutTag_RowMajor : I32EnumAttrCase<"RowMajor", 1>;
def Tla_LayoutTag_ColumnMajor : I32EnumAttrCase<"ColumnMajor", 2>;
def Tla_LayoutTag_zN : I32EnumAttrCase<"zN", 3>;
def Tla_LayoutTag_nZ : I32EnumAttrCase<"nZ", 4>;
def Tla_LayoutTag_L0CLayout : I32EnumAttrCase<"L0Clayout", 5>;
def Tla_LayoutTag_zNUnAlign : I32EnumAttrCase<"zNUnAlign", 6>;
// MX shared-exponent scale block. Same fractal nesting as the "zZ"
// tag, but its C0 is fixed at 2 elements / 32 bytes per fractal by the e8m0
// scale format rather than derived from the element width.
def Tla_LayoutTag_zZMxScale : I32EnumAttrCase<"zZMxScale", 7>;
// The B-side MX scale block uses the transposed nN nesting.
def Tla_LayoutTag_nNMxScale : I32EnumAttrCase<"nNMxScale", 8>;
// GM-side MX scale blocks, named by side (A / B) and orientation. These say how
// the scales sit in *global* memory, which the L1 tags above cannot: a
// pre-swizzled buffer and a natural one are both u8 and otherwise
// indistinguishable, and they select different copies. The four match Catlass's
// isMxScaleFor{Row,Column}Major{A,B} predicates one-to-one; scaleA reaches L1 as
// zZMxScale and scaleB as nNMxScale either way.
def Tla_LayoutTag_RowMajorMxScaleA : I32EnumAttrCase<"RowMajorMxScaleA", 9>;
def Tla_LayoutTag_ColMajorMxScaleA : I32EnumAttrCase<"ColMajorMxScaleA", 10>;
def Tla_LayoutTag_RowMajorMxScaleB : I32EnumAttrCase<"RowMajorMxScaleB", 11>;
def Tla_LayoutTag_ColMajorMxScaleB : I32EnumAttrCase<"ColMajorMxScaleB", 12>;
// A one-dimensional, unit-stride run of elements. This is the tag that matches
// `tla::detail::isVector` on the tile side, so the VectorLayout TileCopyTla
// specializations serve it directly. RowMajor stays two-dimensional on purpose:
// a flat tile says so with this tag rather than overloading RowMajor's shape.
def Tla_LayoutTag_VectorLayout : I32EnumAttrCase<"VectorLayout", 13>;
def Tla_LayoutTagEnum : Tla_I32Enum<"LayoutTag", "Tensor layout tag for tla.tensor types.", [
  Tla_LayoutTag_Unknown,
  Tla_LayoutTag_RowMajor,
  Tla_LayoutTag_ColumnMajor,
  Tla_LayoutTag_zN,
  Tla_LayoutTag_nZ,
  Tla_LayoutTag_L0CLayout,
  Tla_LayoutTag_zNUnAlign,
  Tla_LayoutTag_zZMxScale,
  Tla_LayoutTag_nNMxScale,
  Tla_LayoutTag_RowMajorMxScaleA,
  Tla_LayoutTag_ColMajorMxScaleA,
  Tla_LayoutTag_RowMajorMxScaleB,
  Tla_LayoutTag_ColMajorMxScaleB,
  Tla_LayoutTag_VectorLayout]>;
def Tla_LayoutTagAttr : Tla_I32EnumAttr<"layout_tag", Tla_LayoutTagEnum> {
  let description = [{
    tla layout_tag RowMajor,ColumnMajor,zN,nZ,L0Clayout,zNUnAlign,VectorLayout for generic data types,
    and zZMxScale,nNMxScale,RowMajorMxScaleA,ColMajorMxScaleA,RowMajorMxScaleB,ColMajorMxScaleB for mxfp4/mxfp8 scale.
  }];
}

def Tla_AddressSpace_Generic : I32EnumAttrCase<"generic", 0>;
def Tla_AddressSpace_Gm : I32EnumAttrCase<"gm", 1>;
def Tla_AddressSpace_L1 : I32EnumAttrCase<"l1", 2>;
def Tla_AddressSpace_L0a : I32EnumAttrCase<"l0a", 3>;
def Tla_AddressSpace_L0b : I32EnumAttrCase<"l0b", 4>;
def Tla_AddressSpace_L0c : I32EnumAttrCase<"l0c", 5>;
def Tla_AddressSpace_Ub : I32EnumAttrCase<"ub", 6>;
def Tla_AddressSpace_Biasbuf : I32EnumAttrCase<"biasbuf", 12>;
def Tla_AddressSpaceEnum : Tla_I32Enum<"AddressSpace", "Tla pointer/tensor/memref address space.", [
  Tla_AddressSpace_Generic,
  Tla_AddressSpace_Gm,
  Tla_AddressSpace_L1,
  Tla_AddressSpace_L0a,
  Tla_AddressSpace_L0b,
  Tla_AddressSpace_L0c,
  Tla_AddressSpace_Ub,
  Tla_AddressSpace_Biasbuf
]>;

def Tla_Pipe_Scalar : I32EnumAttrCase<"scalar", 0>;
def Tla_Pipe_Vector : I32EnumAttrCase<"vector", 1>;
def Tla_Pipe_Cube : I32EnumAttrCase<"cube", 2>;
def Tla_Pipe_Mte1 : I32EnumAttrCase<"mte1", 3>;
def Tla_Pipe_Mte2 : I32EnumAttrCase<"mte2", 4>;
def Tla_Pipe_Mte3 : I32EnumAttrCase<"mte3", 5>;
def Tla_Pipe_All : I32EnumAttrCase<"all", 6>;
def Tla_Pipe_Fix : I32EnumAttrCase<"fix", 10>;
def Tla_PipeEnum : Tla_I32Enum<"Pipe", "Pipe identifier for tla.set_flag and tla.wait_flag.", [Tla_Pipe_Scalar, Tla_Pipe_Vector, Tla_Pipe_Cube, Tla_Pipe_Mte1, Tla_Pipe_Mte2, Tla_Pipe_Mte3, Tla_Pipe_All, Tla_Pipe_Fix]>;
def Tla_PipeAttr : Tla_EnumAttr<"Pipe", "pipe"> {
  let parameters = (ins EnumParameter<Tla_PipeEnum>:$pipe);
  let assemblyFormat = "`<` params `>`";
  let description = [{
    Pipe identifier for tla.set_flag and tla.wait_flag.

    Not every value is valid on every op that takes a pipe. The flag and barrier
    ops accept all eight. tla.mutex_lock / tla.mutex_unlock reject all, which has
    no mutex form, and otherwise accept a set that depends on the core their
    enclosing function is classified for: scalar, cube, mte1, mte2, mte3 and fix
    on AIC, and scalar, vector, mte2 and mte3 on AIV -- so only scalar, mte2 and
    mte3 are valid on both. This mirrors the pipes ccec accepts for get_buf /
    rls_buf per core. A rejected pipe is caught in tla-lower-mutex-to-std, not at
    the call site.
  }];
}

// QuantMode Enum Attr
def Tla_QuantMode_NoQuant : I32EnumAttrCase<"NO_QUANT", 0>;
def Tla_QuantMode_PerTensor : I32EnumAttrCase<"PER_TENSOR", 1>;
def Tla_QuantMode_PerChannel : I32EnumAttrCase<"PER_CHANNEL", 2>;
def Tla_QuantModeEnum : Tla_I32Enum<"QuantMode", "Quant Mode when l0c2dst.", [
  Tla_QuantMode_NoQuant,
  Tla_QuantMode_PerTensor,
  Tla_QuantMode_PerChannel
]>;
def Tla_QuantModeAttr : Tla_EnumAttr<"QuantMode", "quant_mode"> {
  let parameters = (ins EnumParameter<Tla_QuantModeEnum>:$quant_mode);
  let assemblyFormat = "`<` params `>`";
  let description = [{
    Copy L0C to Dst quant mode.
  }];
}

// L0C2UBMode Enum Attr
def Tla_L0C2UBMode_NoSplitVec0 : I32EnumAttrCase<"NO_SPLIT_VEC_0", 0>;
def Tla_L0C2UBMode_NoSplitVec1 : I32EnumAttrCase<"NO_SPLIT_VEC_1", 1>;
def Tla_L0C2UBMode_SplitM : I32EnumAttrCase<"SPLIT_M", 2>;
def Tla_L0C2UBMode_SplitN : I32EnumAttrCase<"SPLIT_N", 3>;
def Tla_L0C2UBModeEnum : Tla_I32Enum<"L0C2UBMode", "L0C 2 UB split mode.", [
  Tla_L0C2UBMode_NoSplitVec0,
  Tla_L0C2UBMode_NoSplitVec1,
  Tla_L0C2UBMode_SplitM,
  Tla_L0C2UBMode_SplitN
]>;
def Tla_L0C2UBModeAttr : Tla_EnumAttr<"L0C2UBMode", "l0c2ub_mode"> {
  let parameters = (ins EnumParameter<Tla_L0C2UBModeEnum>:$l0c2ub_mode);
  let assemblyFormat = "`<` params `>`";
  let description = [{
    Copy L0C 2 UB split mode.
  }];
}

// LoadDist Enum Attr
def Tla_LoadDist_Norm : I32EnumAttrCase<"norm", 0>;
def Tla_LoadDist_BrcB32 : I32EnumAttrCase<"brc_b32", 1>;
def Tla_LoadDist_DintlvB32 : I32EnumAttrCase<"dintlv_b32", 2>;
def Tla_LoadDist_UsB8 : I32EnumAttrCase<"us_b8", 3>;
def Tla_LoadDist_BrcB16 : I32EnumAttrCase<"brc_b16", 4>;
def Tla_LoadDist_UsB16 : I32EnumAttrCase<"us_b16", 5>;
// MaskDist US/DS (free slots; upstream skipped 6 between us_b16 and unpack_b16).
def Tla_LoadDist_Us : I32EnumAttrCase<"us", 6>;
def Tla_LoadDist_UnpackB16 : I32EnumAttrCase<"unpack_b16", 7>;
def Tla_LoadDist_E2bB16 : I32EnumAttrCase<"e2b_b16", 8>;
def Tla_LoadDist_E2bB32 : I32EnumAttrCase<"e2b_b32", 9>;
def Tla_LoadDist_Blk : I32EnumAttrCase<"blk", 10>;
def Tla_LoadDist_Ds : I32EnumAttrCase<"ds", 11>;
def Tla_LoadDistEnum : Tla_I32Enum<"LoadDist", "Load distribution for tla.load.", [
  Tla_LoadDist_Norm,
  Tla_LoadDist_BrcB32,
  Tla_LoadDist_DintlvB32,
  Tla_LoadDist_UsB8,
  Tla_LoadDist_BrcB16,
  Tla_LoadDist_UsB16,
  Tla_LoadDist_Us,
  Tla_LoadDist_UnpackB16,
  Tla_LoadDist_E2bB16,
  Tla_LoadDist_E2bB32,
  Tla_LoadDist_Blk,
  Tla_LoadDist_Ds
]>;
def Tla_LoadDistAttr : Tla_EnumAttr<"LoadDist", "load_dist"> {
  let parameters = (ins EnumParameter<Tla_LoadDistEnum>:$load_dist);
  let assemblyFormat = "`<` params `>`";
  let description = [{
    Broadcast/load distribution pattern for tla.load.
    ``dintlv_b32`` is dual-destination de-interleave load (even/odd b32
    elements into two vector registers); requires a second result.
    ``us_b8`` is single-destination 2x up-sample load (AscendC
    ``LoadDist::DIST_US_B8``): loads ``VL/2`` b8 elements and repeats each
    twice to fill a VL-wide ``b8`` register; b8 element type only.
    ``brc_b16`` is single-destination broadcast load (AscendC
    ``LoadDist::DIST_BRC_B16``): reads one b16 element and replicates it
    across all lanes of a VL-wide ``b16`` register; b16 element type only.
    ``us_b16`` is single-destination 2x up-sample load (AscendC
    ``LoadDist::DIST_US_B16``): loads ``VL/2`` b16 elements and repeats
    each twice to fill a VL-wide ``b16`` register; b16 element type only.
    ``us`` / ``ds`` are AscendC ``MaskDist::DIST_US`` / ``DIST_DS`` for
    ``!tla.mask`` loads (``hivm.plds.b8`` dist 1 / 2).
    ``unpack_b16`` is single-destination unpack load (AscendC
    ``LoadDist::DIST_UNPACK_B16``, HIVMAVE ``UNPK_B16``): loads ``VL/2``
    b16 elements and inserts one zero b16 element after each to fill a
    VL-wide ``b16`` register; b16 element type only.
    ``e2b_b16`` / ``e2b_b32`` are element-to-DataBlock broadcast loads (AscendC
    ``LoadDist::DIST_E2B_B16`` / ``DIST_E2B_B32``, HIVMAVE ``E2B_B16`` /
    ``E2B_B32``): reads ``VL/DB`` (= 8, with ``DB`` a 32-byte DataBlock)
    elements from the tile view base and replicates each element across all
    lanes of its destination DataBlock to fill a VL-wide register;
    ``e2b_b16`` requires a 2-byte element type (f16/bf16/i16/u16, 16 lanes
    per DataBlock), ``e2b_b32`` a 4-byte element type (f32/i32/u32, 8 lanes
    per DataBlock).
    ``blk`` is a DataBlock broadcast load (AscendC ``LoadDist::DIST_BLK``,
    HIVMAVE ``BLK``): reads the ``32-byte`` DataBlock (``DB``) worth of
    elements at the tile view base (``b8``: 32, ``b16``: 16, ``b32``: 8
    elements) and replicates it across all ``VL/DB`` (= 8) DataBlocks to
    fill a VL-wide register; works for any element size.
  }];
}

// StoreDist Enum Attr
def Tla_StoreDist_Norm : I32EnumAttrCase<"norm", 0>;
def Tla_StoreDist_PackB32: I32EnumAttrCase<"pack_b32", 1>;
def Tla_StoreDist_PackB16: I32EnumAttrCase<"pack_b16", 2>;
def Tla_StoreDist_FirstElementB8 : I32EnumAttrCase<"first_element_b8", 3>;
def Tla_StoreDist_FirstElementB16 : I32EnumAttrCase<"first_element_b16", 4>;
def Tla_StoreDist_FirstElementB32 : I32EnumAttrCase<"first_element_b32", 5>;
def Tla_StoreDistEnum : Tla_I32Enum<"StoreDist", "Store distribution for tla.store.", [
  Tla_StoreDist_Norm,
  Tla_StoreDist_PackB32,
  Tla_StoreDist_PackB16,
  Tla_StoreDist_FirstElementB8,
  Tla_StoreDist_FirstElementB16,
  Tla_StoreDist_FirstElementB32
]>;
def Tla_StoreDistAttr : Tla_EnumAttr<"StoreDist", "store_dist"> {
  let parameters = (ins EnumParameter<Tla_StoreDistEnum>:$store_dist);
  let assemblyFormat = "`<` params `>`";
  let description = [{
    Store distribution for tla.store.
     - DIST_NORM: Normal mode, support b8/b16/b32, with a fixed datacopy VL, should be aligned with 32B.
     - DIST_PACK_B16: Compact mode, store the low half data (b16) to dst, should be aligned with 32B, or VL/2.
     - DIST_PACK_B32: Compact mode, store the low half data (b32) to dst.
     - DIST_FIRST_ELEMENT_B8/B16/B32 (AscendC ``DIST_FIRST_ELEMENT_*`` / HIVMAVE
       ``ONEPT_B*``): element store that ignores the mask and writes only the
       first element (lane 0) of the source register to the dst base address
       (AscendC ``asc_storealign_1st``). The ``_b8``/``_b16``/``_b32`` suffix
       fixes the stored element width in bytes and must match both the source
       and the dst element type; the dst base must stay 32B-aligned.
  }];
}

// ---------------------------------------------------------------------------
// CastParams: the four knobs of a tla.cast, matching CastParams in the Python
// front-end (catlass/params.py). The enum keywords are the MLIR-asm tokens.
// ---------------------------------------------------------------------------

// RegSlot: which half of a packed register a narrow result lands in
// 2x casts use zero=part_even / one=part_odd; 4x casts (i32<->i8) map all four
// values onto the pack pattern zero=pp0 / one=pp1 / two=pp2 / three=pp3.
def Tla_RegSlot_Zero : I32EnumAttrCase<"zero", 0>;
def Tla_RegSlot_One : I32EnumAttrCase<"one", 1>;
def Tla_RegSlot_Two : I32EnumAttrCase<"two", 2>;
def Tla_RegSlot_Three : I32EnumAttrCase<"three", 3>;
def Tla_RegSlotEnum : Tla_I32Enum<"RegSlot", "Packed-register position for a narrow cast result.", [
  Tla_RegSlot_Zero,
  Tla_RegSlot_One,
  Tla_RegSlot_Two,
  Tla_RegSlot_Three
]>;

// ---------------------------------------------------------------------------
// AtomicMode: the atomic mode passing to ``tla.params.CopyParams``
// front-end (catlass/params.py). The enum keywords are the MLIR-asm tokens.
// ---------------------------------------------------------------------------

// AtomicMode: None, Add

def Tla_AtomicMode_None : I32EnumAttrCase<"none", 0>;
def Tla_AtomicMode_Add : I32EnumAttrCase<"add", 1>;

def Tla_AtomicMode_Enum : Tla_I32Enum<"AtomicMode", "Atomic type", [
  Tla_AtomicMode_None,
  Tla_AtomicMode_Add
]>;

def Tla_AtomicModeAttr : Tla_EnumAttr<"AtomicMode", "atomic_mode"> {
  let parameters = (ins EnumParameter<Tla_AtomicMode_Enum>:$atomic_mode);
  let assemblyFormat = "`<` params `>`";
  let description = [{
    Atomic operation kind for tla.copy

    The supported atomic mode listed below:
      - ADD
  }];
}

// ---------------------------------------------------------------------------
// Histograms: FREQUENCY / ACCUMULATE + BIN0 / BIN1
// ---------------------------------------------------------------------------

def Tla_HistogramsType_Frequency : I32EnumAttrCase<"frequency", 0>;
def Tla_HistogramsType_Accumulate : I32EnumAttrCase<"accumulate", 1>;

def Tla_HistogramsType_Enum : Tla_I32Enum<"HistogramsType",
    "Reg histogram type (FREQUENCY / ACCUMULATE)", [
  Tla_HistogramsType_Frequency,
  Tla_HistogramsType_Accumulate
]>;

def Tla_HistogramsTypeAttr : Tla_EnumAttr<"HistogramsType", "histograms_type"> {
  let parameters = (ins EnumParameter<Tla_HistogramsType_Enum>:$histograms_type);
  let assemblyFormat = "`<` params `>`";
}

def Tla_HistogramsBin_Bin0 : I32EnumAttrCase<"bin0", 0>;
def Tla_HistogramsBin_Bin1 : I32EnumAttrCase<"bin1", 1>;

def Tla_HistogramsBin_Enum : Tla_I32Enum<"HistogramsBin",
    "Reg histogram bin half (BIN0 / BIN1)", [
  Tla_HistogramsBin_Bin0,
  Tla_HistogramsBin_Bin1
]>;

def Tla_HistogramsBinAttr : Tla_EnumAttr<"HistogramsBin", "histograms_bin"> {
  let parameters = (ins EnumParameter<Tla_HistogramsBin_Enum>:$histograms_bin);
  let assemblyFormat = "`<` params `>`";
}

// ComputeOrder: mmad M/N compute-direction priority (CTRL[51]).
// M_FIRST(0) is the hardware default; N_FIRST(1) flips the priority. Lowered to
// hivm::SetCtrlOp in TlaCubeRegionPass; absence keeps the hardware default.
def Tla_ComputeOrder_MFirst : I32EnumAttrCase<"M_FIRST", 0>;
def Tla_ComputeOrder_NFirst : I32EnumAttrCase<"N_FIRST", 1>;
def Tla_ComputeOrder_Enum : Tla_I32Enum<"ComputeOrder",
  "mmad compute order (M/N direction priority)", [
  Tla_ComputeOrder_MFirst, Tla_ComputeOrder_NFirst
]>;
def Tla_ComputeOrderAttr : Tla_EnumAttr<"ComputeOrder", "compute_order"> {
  let parameters = (ins EnumParameter<Tla_ComputeOrder_Enum>:$value);
  let assemblyFormat = "`<` params `>`";
  let description = [{
    mmad compute-direction priority, lowered to SPR.CTRL[51]:

      - M_FIRST (0): M-direction priority (hardware default)
      - N_FIRST (1): N-direction priority
  }];
}

// HF32Mode: mmad HF32 rounding mode (CTRL[46] enable, CTRL[47] rounding).
// HF32_DISABLE(0) is the hardware default (no HF32 rounding).
def Tla_HF32Mode_Disable : I32EnumAttrCase<"HF32_DISABLE", 0>;
def Tla_HF32Mode_NearestZero : I32EnumAttrCase<"HF32_NEAREST_ZERO", 1>;
def Tla_HF32Mode_NearestEven : I32EnumAttrCase<"HF32_NEAREST_EVEN", 2>;
def Tla_HF32Mode_Enum : Tla_I32Enum<"HF32Mode",
  "mmad HF32 rounding mode", [
  Tla_HF32Mode_Disable, Tla_HF32Mode_NearestZero, Tla_HF32Mode_NearestEven
]>;
def Tla_HF32ModeAttr : Tla_EnumAttr<"HF32Mode", "hf32_mode"> {
  let parameters = (ins EnumParameter<Tla_HF32Mode_Enum>:$value);
  let assemblyFormat = "`<` params `>`";
  let description = [{
    mmad HF32 rounding mode, lowered to SPR.CTRL[46]/CTRL[47]:

      - HF32_DISABLE (0): HF32 mode disabled (hardware default)
      - HF32_NEAREST_ZERO (1): f32 rounded to HF32 towards zero
      - HF32_NEAREST_EVEN (2): f32 rounded to HF32 to nearest even
  }];
}

// SatMode: overflow behaviour of the cast (AVE `sat` BoolAttr).
def Tla_SatMode_Unknown : I32EnumAttrCase<"unknown", 0>;
def Tla_SatMode_Sat : I32EnumAttrCase<"sat", 1>;
def Tla_SatMode_Nosat : I32EnumAttrCase<"nosat", 2>;
def Tla_SatModeEnum : Tla_I32Enum<"SatMode", "Cast saturation behaviour on overflow.", [
  Tla_SatMode_Unknown,
  Tla_SatMode_Sat,
  Tla_SatMode_Nosat
]>;

// RoundMode: rounding applied on precision loss (HIVM round_mode).
// Codes 0-3 are frozen ABI for existing IR; 4-6 append AscendC CAST_NONE /
// CAST_RINT / CAST_ODD. CAST_HYBRID is intentionally omitted.
// CAST_NONE is a policy: lossless casts drop the round operand; lossy casts
// lower as RINT (ties to even), matching AscendC.
def Tla_RoundMode_CastRound : I32EnumAttrCase<"cast_round", 0>;
def Tla_RoundMode_CastFloor : I32EnumAttrCase<"cast_floor", 1>;
def Tla_RoundMode_CastCeil : I32EnumAttrCase<"cast_ceil", 2>;
def Tla_RoundMode_CastTrunc : I32EnumAttrCase<"cast_trunc", 3>;
def Tla_RoundMode_CastNone : I32EnumAttrCase<"cast_none", 4>;
def Tla_RoundMode_CastRint : I32EnumAttrCase<"cast_rint", 5>;
def Tla_RoundMode_CastOdd : I32EnumAttrCase<"cast_odd", 6>;
def Tla_RoundModeEnum : Tla_I32Enum<"RoundMode", "Cast rounding mode on precision loss.", [
  Tla_RoundMode_CastRound,
  Tla_RoundMode_CastFloor,
  Tla_RoundMode_CastCeil,
  Tla_RoundMode_CastTrunc,
  Tla_RoundMode_CastNone,
  Tla_RoundMode_CastRint,
  Tla_RoundMode_CastOdd
]>;

// The three cast knobs are carried on tla.cast as a builtin DenseI32ArrayAttr of
// three enum codes, in this fixed order (see catlass/params.py):
//   [0] reg_slot (RegSlot)   [1] sat_mode (SatMode)   [2] round_mode (RoundMode)
// A builtin array attr round-trips textually through every pass; the enum
// definitions above give the C++ enum types the lowering casts the codes to.

def Tla_RegPart_Low : I32EnumAttrCase<"low", 0>;
def Tla_RegPart_High : I32EnumAttrCase<"high", 1>;
def Tla_RegPartEnum : Tla_I32Enum<
    "RegPart", "Register half selection.", [
      Tla_RegPart_Low,
      Tla_RegPart_High
    ]>;
def Tla_RegPartAttr : Tla_I32EnumAttr<"reg_part", Tla_RegPartEnum> {
  let description = [{
    Selects the low or high half of a register.
  }];
}
