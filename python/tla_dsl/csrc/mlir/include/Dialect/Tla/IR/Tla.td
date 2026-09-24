include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/CommonTypeConstraints.td"
include "mlir/IR/DialectBase.td"
include "mlir/IR/EnumAttr.td"
include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
// include "Dialect/Tla/IR/TlaOpTraits.h"
// SameOperandsAndResultShape
// let hasFolder = 1;

def TlaDialect : Dialect {
  let name = "tla";
  let summary = "Tla dialect schema for TlaIR matmul prototyping.";
  let description = [{
    Tla dialect schema for TlaIR matmul prototyping.
  }];
  let cppNamespace = "::tla";
  // auto generate parseType/printType/parseAttribute/printAttribute
  let useDefaultAttributePrinterParser = 1;
  let useDefaultTypePrinterParser = 1;
}

class Tla_Op<string mnemonic, list<Trait> traits = []> : Op<TlaDialect, mnemonic, traits>;

def Tla_CrossFlagType : TypeDef<TlaDialect, "CrossFlag", []> {
  let mnemonic = "cross_flag";
  let summary = "Synchronization flag used for cross-core coordination.";
  let parameters = (ins "int64_t":$mode);
  let assemblyFormat = "`<` $mode `>`";
}

def Tla_FlagType : TypeDef<TlaDialect, "Flag", []> {
  let mnemonic = "flag";
  let summary = "Synchronization flag used for pipe coordination.";
  let assemblyFormat = "";
}

def Tla_MaskSSAType : TypeDef<TlaDialect, "MaskSSA", []> {
  let mnemonic = "mask";
  let summary = "Register-resident predicate mask SSA value.";
  let parameters = (ins "int64_t":$physicalLanes);
  let assemblyFormat = "`<` $physicalLanes `>`";
  let genVerifyDecl = 1;
}


def Tla_MutexType : TypeDef<TlaDialect, "Mutex", []> {
  let mnemonic = "mutex";
  let summary = "Synchronization mutex used for resource coordination.";
  let assemblyFormat = "";
}


def Tla_AddressSpaceTypeParameter
    : TypeParameter<"::AddressSpace", "Tla address-space keyword"> {
  let parser = [{ [&]() -> ::mlir::FailureOr<::AddressSpace> {
    ::llvm::StringRef kw;
    if ($_parser.parseKeyword(&kw))
      return ::mlir::failure();
    auto sym = ::symbolizeAddressSpace(kw);
    if (!sym)
      return $_parser.emitError($_parser.getCurrentLocation())
             << "unknown AddressSpace keyword: " << kw;
    return *sym;
  }() }];
  let printer = "$_printer << ::stringifyAddressSpace($_self)";
}

class Tla_IndexTreeType<string name, string typeMnemonic, string typeSummary>
    : TypeDef<TlaDialect, name, []> {
  let mnemonic = typeMnemonic;
  let summary = typeSummary;
  let parameters = (ins ArrayRefParameter<"int64_t", "encoded index-tree">:$tree);
  let hasCustomAssemblyFormat = 1;
  let genVerifyDecl = 1;
}

def Tla_ShapeType : Tla_IndexTreeType<"Shape", "shape", "Shape value for tile/broadcast.">;

def Tla_CoordType : Tla_IndexTreeType<"Coord", "coord", "Coordinate tuple for hierarchical tiling.">;

def Tla_StrideType : Tla_IndexTreeType<"Stride", "stride", "Stride tuple for layouts.">;

def Tla_VectorSSAType : TypeDef<TlaDialect, "VectorSSA", []> {
  let mnemonic = "vector";
  let summary = "Register-resident vector SSA value.";
  let parameters = (ins
    "int64_t":$validLanes,
    "::mlir::Type":$elementType
  );
  let hasCustomAssemblyFormat = 1;
  let genVerifyDecl = 1;
}

def Tla_ScalarValueType : AnyTypeOf<[AnyInteger, AnyFloat, Index],
    "TLA scalar SSA value type">;

def Tla_IntegerOrIndexType : AnyTypeOf<[AnyInteger, Index],
    "TLA integer-or-index SSA value type">;

def Tla_BoolAttrAttr : DialectAttr<TlaDialect, CPred<"true">, "Builtin boolean attribute placeholder.">;

def Tla_I64AttrAttr : DialectAttr<TlaDialect, CPred<"true">, "Builtin i64 attribute placeholder.">;

def Tla_StrAttrAttr : DialectAttr<TlaDialect, CPred<"true">, "Builtin string attribute placeholder.">;

def Tla_SymbolNameAttrAttr : DialectAttr<TlaDialect, CPred<"true">, "Builtin symbol name attribute placeholder.">;

def Tla_TypeAttrAttr : DialectAttr<TlaDialect, CPred<"true">, "Builtin type attribute placeholder.">;

class Tla_EnumAttr<string name, string attrMnemonic, list<Trait> traits = []>
    : AttrDef<TlaDialect, name, traits> {
  let mnemonic = attrMnemonic;
}

class Tla_I32Enum<string name, string description, list<I32EnumAttrCase> cases>
    : I32EnumAttr<name, description, cases> {
}

class Tla_I32EnumAttr<string attrMnemonic, Tla_I32Enum enumInfo>
    : EnumAttr<TlaDialect, enumInfo, attrMnemonic> {
  let assemblyFormat = "`<` $value `>`";
  let cppNamespace = "::tla";
}

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
  Tla_LayoutTag_ColMajorMxScaleB]>;
def Tla_LayoutTagAttr : Tla_I32EnumAttr<"layout_tag", Tla_LayoutTagEnum> {
  let description = [{
    tla layout_tag RowMajor,ColumnMajor,zN,nZ,L0Clayout,zNUnAlign for generic data types,
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
def Tla_AddressSpaceEnum : Tla_I32Enum<"AddressSpace", "Tla pointer/tensor/memref address space.", [Tla_AddressSpace_Generic, Tla_AddressSpace_Gm, Tla_AddressSpace_L1, Tla_AddressSpace_L0a, Tla_AddressSpace_L0b, Tla_AddressSpace_L0c, Tla_AddressSpace_Ub]>;

def Tla_PtrAddressSpaceAttr : Tla_EnumAttr<"PtrAddressSpace", "ptr_address_space"> {
  let parameters = (ins EnumParameter<Tla_AddressSpaceEnum>:$value);
  let assemblyFormat = "`<` $value `>`";
  let description = [{
    On-chip ``!tla.ptr`` address space; carried on pointer types (e.g. with ``tla.alloc_ptr``).
  }];
}


// Cube element formats that this MLIR has no builtin type for.
//
// LLVM 19 ships Float8E4M3FN / Float8E5M2 and nothing narrower: Float4E2M1FN and
// Float8E8M0FNU both arrive in LLVM 20. Spelling them as dialect types rather
// than borrowing i4 / i8 keeps the *format* in the type, where it belongs --
// e2m1 and e1m2 are the same width and differ only in exponent/mantissa split,
// so a shared integer carrier cannot tell them apart and the encoding has to be
// smuggled alongside on an attribute that nothing verifies against the type.
//
// These are storage-only: no arithmetic is ever performed on them in MLIR (the
// cube's mad_mx does it), so they need no float semantics. At the memref
// boundary they are buffered as i8 -- see storageTypeAndCount -- which is what
// keeps every byte-denominated thing downstream meaning bytes.
//
// When the toolchain reaches LLVM 20 these become aliases for the builtin
// Float4E2M1FN / Float8E8M0FNU and nothing else has to change.
class Tla_CubeElementType<string name, string typeMnemonic, string desc>
    : TypeDef<TlaDialect, name, []> {
  let mnemonic = typeMnemonic;
  let summary = desc;
}

def Tla_Float4E2M1Type : Tla_CubeElementType<"Float4E2M1", "f4e2m1",
    "Packed 4-bit float, 2-bit exponent / 1-bit mantissa (OCP MXFP4).">;
def Tla_Float4E1M2Type : Tla_CubeElementType<"Float4E1M2", "f4e1m2",
    "Packed 4-bit float, 1-bit exponent / 2-bit mantissa.">;
def Tla_Float8E8M0Type : Tla_CubeElementType<"Float8E8M0", "f8e8m0",
    "8-bit shared exponent (OCP microscaling scale block).">;

def Tla_PtrType : TypeDef<TlaDialect, "Ptr", []> {
  let summary = "First-class pointer SSA value (pointee, address space, alignment).";
  let description = [{
    A logical pointer that can flow through function calls, branches, and SCF
    region arguments/results like any other SSA value. ``tla-lower-ptr`` converts
    structural pointer carriers atomically to ``i64`` byte addresses.
  }];
  let mnemonic = "ptr";
  let parameters = (ins
    TypeParameter<"::mlir::Type", "Logical pointee / value element type.">:$pointee,
    Tla_AddressSpaceTypeParameter:$addrspace,
    "unsigned":$alignment
  );
  let assemblyFormat = "`<` $pointee `,` $addrspace `,` $alignment `>`";
}

def Tla_LayoutType : TypeDef<TlaDialect, "Layout", []> {
  let mnemonic = "layout";
  let summary = "Packed shape+stride+layout-tag descriptor (Tla-style).";
  let parameters = (ins
    TypeParameter<"::tla::ShapeType", "shape tree type">:$shape,
    TypeParameter<"::tla::StrideType", "stride tree type">:$stride,
    TypeParameter<"::tla::ShapeType", "origin shape">:$origin,
    EnumParameter<Tla_LayoutTagEnum>:$layoutTag
  );
  let assemblyFormat = "`<` qualified($shape) `,` qualified($stride) `,` qualified($origin) `,` $layoutTag `>`";
  let genVerifyDecl = 1;
}

def Tla_TensorType : TypeDef<TlaDialect, "TlaTensor", []> {
  let summary = "First-class tensor/view with structured layout, coordinate, and pointer.";
  let description = [{
    Structured tensor descriptor:
    ``!tla.tensor<!tla.layout<...>, !tla.coord<...>, !tla.ptr<...>>``.
  }];
  let mnemonic = "tensor";
  let parameters = (ins
    TypeParameter<"::tla::LayoutType", "layout descriptor">:$layout,
    TypeParameter<"::tla::CoordType", "coord tree type">:$coord,
    TypeParameter<"::tla::PtrType", "backing pointer type">:$ptr
  );
  let assemblyFormat = "`<` qualified($layout) `,` qualified($coord) `,` qualified($ptr) `>`";
  let genVerifyDecl = 1;
}

def Tla_Pipe_Scalar : I32EnumAttrCase<"scalar", 0>;
def Tla_Pipe_Vector : I32EnumAttrCase<"vector", 1>;
def Tla_Pipe_Cube : I32EnumAttrCase<"cube", 2>;
def Tla_Pipe_Mte1 : I32EnumAttrCase<"mte1", 3>;
def Tla_Pipe_Mte2 : I32EnumAttrCase<"mte2", 4>;
def Tla_Pipe_Mte3 : I32EnumAttrCase<"mte3", 5>;
def Tla_Pipe_All : I32EnumAttrCase<"all", 6>;
def Tla_Pipe_Mte4 : I32EnumAttrCase<"mte4", 7>;
def Tla_Pipe_Mte5 : I32EnumAttrCase<"mte5", 8>;
def Tla_Pipe_V2 : I32EnumAttrCase<"v2", 9>;
def Tla_Pipe_Fix : I32EnumAttrCase<"fix", 10>;
def Tla_Pipe_VirtualMte2L1a : I32EnumAttrCase<"virtual_mte2_l1a", 11>;
def Tla_Pipe_VirtualMte2L1b : I32EnumAttrCase<"virtual_mte2_l1b", 12>;
def Tla_Pipe_Num : I32EnumAttrCase<"num", 13>;
def Tla_PipeEnum : Tla_I32Enum<"Pipe", "Pipe identifier for tla.set_flag and tla.wait_flag.", [Tla_Pipe_Scalar, Tla_Pipe_Vector, Tla_Pipe_Cube, Tla_Pipe_Mte1, Tla_Pipe_Mte2, Tla_Pipe_Mte3, Tla_Pipe_All, Tla_Pipe_Mte4, Tla_Pipe_Mte5, Tla_Pipe_V2, Tla_Pipe_Fix, Tla_Pipe_VirtualMte2L1a, Tla_Pipe_VirtualMte2L1b, Tla_Pipe_Num]>;
def Tla_PipeAttr : Tla_EnumAttr<"Pipe", "pipe"> {
  let parameters = (ins EnumParameter<Tla_PipeEnum>:$pipe);
  let assemblyFormat = "`<` params `>`";
  let description = [{
    Pipe identifier for tla.set_flag and tla.wait_flag.
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
def Tla_LoadDist_UnpackB16 : I32EnumAttrCase<"unpack_b16", 7>;
def Tla_LoadDist_E2bB16 : I32EnumAttrCase<"e2b_b16", 8>;
def Tla_LoadDist_E2bB32 : I32EnumAttrCase<"e2b_b32", 9>;
def Tla_LoadDist_Blk : I32EnumAttrCase<"blk", 10>;
def Tla_LoadDistEnum : Tla_I32Enum<"LoadDist", "Load distribution for tla.load.", [
  Tla_LoadDist_Norm,
  Tla_LoadDist_BrcB32,
  Tla_LoadDist_DintlvB32,
  Tla_LoadDist_UsB8,
  Tla_LoadDist_BrcB16,
  Tla_LoadDist_UsB16,
  Tla_LoadDist_UnpackB16,
  Tla_LoadDist_E2bB16,
  Tla_LoadDist_E2bB32,
  Tla_LoadDist_Blk
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
def Tla_RoundMode_CastRound : I32EnumAttrCase<"cast_round", 0>;
def Tla_RoundMode_CastFloor : I32EnumAttrCase<"cast_floor", 1>;
def Tla_RoundMode_CastCeil : I32EnumAttrCase<"cast_ceil", 2>;
def Tla_RoundMode_CastTrunc : I32EnumAttrCase<"cast_trunc", 3>;
def Tla_RoundModeEnum : Tla_I32Enum<"RoundMode", "Cast rounding mode on precision loss.", [
  Tla_RoundMode_CastRound,
  Tla_RoundMode_CastFloor,
  Tla_RoundMode_CastCeil,
  Tla_RoundMode_CastTrunc
]>;

// The three cast knobs are carried on tla.cast as a builtin DenseI32ArrayAttr of
// three enum codes, in this fixed order (see catlass/params.py):
//   [0] reg_slot (RegSlot)   [1] sat_mode (SatMode)   [2] round_mode (RoundMode)
// A builtin array attr round-trips textually through every pass; the enum
// definitions above give the C++ enum types the lowering casts the codes to.

def Tla_CopyL0C2DstParamsType : TypeDef<TlaDialect, "CopyL0C2DstParams", []> {
  let mnemonic = "copy_l0c2dst_params";
  let summary = "CopyL0C2DstParams for tla.copy.";
  let assemblyFormat = "";
}

def Tla_BinaryOpsTrait : NativeOpTrait<"::mlir::tla::trait::BinaryOps">;
def Tla_BlockDimTrait : NativeOpTrait<"::mlir::tla::trait::BlockDim">;
def Tla_BlockIdxTrait : NativeOpTrait<"::mlir::tla::trait::BlockIdx">;
def Tla_ControlOpsTrait : NativeOpTrait<"::mlir::tla::trait::ControlOps">;
def Tla_CopyTrait : NativeOpTrait<"::mlir::tla::trait::Copy">;
def Tla_CrossCoreSetFlagTrait : NativeOpTrait<"::mlir::tla::trait::CrossCoreSetFlag">;
def Tla_CrossCoreWaitFlagTrait : NativeOpTrait<"::mlir::tla::trait::CrossCoreWaitFlag">;
def Tla_CrossFlagTrait : NativeOpTrait<"::mlir::tla::trait::CrossFlag">;
def Tla_CubeTrait : NativeOpTrait<"::mlir::tla::trait::Cube">;
def Tla_CubeTernaryOpsTrait : NativeOpTrait<"::mlir::tla::trait::CubeTernaryOps">;
def Tla_DependentBinaryOpsTrait : NativeOpTrait<"::mlir::tla::trait::DependentBinaryOps">;
def Tla_DependentUnaryOpsTrait : NativeOpTrait<"::mlir::tla::trait::DependentUnaryOps">;
def Tla_FlagTrait : NativeOpTrait<"::mlir::tla::trait::Flag">;
def Tla_IntermediateBinaryOpsTrait : NativeOpTrait<"::mlir::tla::trait::IntermediateBinaryOps">;
def Tla_IntermediateMemoryOpsTrait : NativeOpTrait<"::mlir::tla::trait::IntermediateMemoryOps">;
def Tla_IntermediateUnaryOpsTrait : NativeOpTrait<"::mlir::tla::trait::IntermediateUnaryOps">;
def Tla_LoadTrait : NativeOpTrait<"::mlir::tla::trait::Load">;
def Tla_MakeTileLikeTrait : NativeOpTrait<"::mlir::tla::trait::MakeTileLike">;
def Tla_MemoryOpsTrait : NativeOpTrait<"::mlir::tla::trait::MemoryOps">;
def Tla_AllocateTrait : NativeOpTrait<"::mlir::tla::trait::Allocate">;
def Tla_MmadTrait : NativeOpTrait<"::mlir::tla::trait::Mmad">;
def Tla_OpClassBinaryTrait : NativeOpTrait<"::mlir::tla::trait::OpClassBinary">;
def Tla_OpClassControlTrait : NativeOpTrait<"::mlir::tla::trait::OpClassControl">;
def Tla_OpClassMemoryTrait : NativeOpTrait<"::mlir::tla::trait::OpClassMemory">;
def Tla_OpClassUnaryTrait : NativeOpTrait<"::mlir::tla::trait::OpClassUnary">;
def Tla_PipeBarrierTrait : NativeOpTrait<"::mlir::tla::trait::PipeBarrier">;
def Tla_PipeCubeOpsTrait : NativeOpTrait<"::mlir::tla::trait::PipeCubeOps">;
def Tla_PipeDependentOpsTrait : NativeOpTrait<"::mlir::tla::trait::PipeDependentOps">;
def Tla_PipeIntermediateOpsTrait : NativeOpTrait<"::mlir::tla::trait::PipeIntermediateOps">;
def Tla_PipeScalarOpsTrait : NativeOpTrait<"::mlir::tla::trait::PipeScalarOps">;
def Tla_PipeVectorOpsTrait : NativeOpTrait<"::mlir::tla::trait::PipeVectorOps">;
def Tla_RegionControlOpsTrait : NativeOpTrait<"::mlir::tla::trait::RegionControlOps">;
def Tla_RegionOpsTrait : NativeOpTrait<"::mlir::tla::trait::RegionOps">;
def Tla_ScalarControlOpsTrait : NativeOpTrait<"::mlir::tla::trait::ScalarControlOps">;
def Tla_SetFlagTrait : NativeOpTrait<"::mlir::tla::trait::SetFlag">;
def Tla_StoreTrait : NativeOpTrait<"::mlir::tla::trait::Store">;
def Tla_SubtileTrait : NativeOpTrait<"::mlir::tla::trait::Subtile">;
def Tla_TernaryOpsTrait : NativeOpTrait<"::mlir::tla::trait::TernaryOps">;
def Tla_TileTrait : NativeOpTrait<"::mlir::tla::trait::Tile">;
def Tla_TileCountTrait : NativeOpTrait<"::mlir::tla::trait::TileCount">;
def Tla_UnaryOpsTrait : NativeOpTrait<"::mlir::tla::trait::UnaryOps">;
def Tla_VectorTrait : NativeOpTrait<"::mlir::tla::trait::Vector">;
def Tla_VectorBinaryOpsTrait : NativeOpTrait<"::mlir::tla::trait::VectorBinaryOps">;
def Tla_WaitFlagTrait : NativeOpTrait<"::mlir::tla::trait::WaitFlag">;

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
// select -- so for a bf16 operand tla-vector-region evaluates these in f32 and
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

def Tla_IntToPtrOp : Tla_Op<"inttoptr", [Pure]> {
  let summary = "Integer SSA to tla.ptr (``tla.inttoptr`` parity)";
  let description = [{
    Reinterprets an integer-like SSA address as the given ``!tla.ptr`` type.
    ``tla-lower-ptr`` normally converts it back to an ``i64`` byte address; an
    instance may remain temporarily at a ``tla.make_tensor{,_like}`` boundary so
    tensor lowering can construct a consumer-local memref descriptor.
  }];
  let arguments = (ins AnyType:$addr);
  let results = (outs Tla_PtrType:$ptr);
  let assemblyFormat = [{
    $addr attr-dict `:` type($addr) `->` qualified(type($ptr))
  }];
}

def Tla_RecastPtrOp : Tla_Op<"recast_ptr", [Pure]> {
  let summary = "Recast pointer element type (``tla.recast_ptr`` parity, dtype-only)";
  let description = [{
    Reinterprets a ``!tla.ptr`` SSA as a different logical pointee type while preserving
    address space and alignment.
    **No swizzle** metadata (Tla does not model swizzle on ``!tla.ptr``).
  }];
  let arguments = (ins Tla_PtrType:$src);
  let results = (outs Tla_PtrType:$ptr);
  let assemblyFormat = [{
    $src attr-dict `:` qualified(type($src)) `->` qualified(type($ptr))
  }];
}

def Tla_TensorPtrOp : Tla_Op<"tensor_ptr", [Pure]> {
  let summary = "Extract the backing pointer from a tla.tensor";
  let description = [{
    Yields the ``!tla.ptr`` SSA value backing a ``!tla.tensor``. Kernel arguments
    are represented inside the function by a root ``tla.tensor_desc``.
  }];
  let arguments = (ins Tla_TensorType:$src);
  let results = (outs Tla_PtrType:$ptr);
  let assemblyFormat = [{
    $src attr-dict `:` qualified(type($src)) `->` qualified(type($ptr))
  }];
  let hasVerifier = 1;
}

def Tla_PtrAddOp : Tla_Op<"ptr_add", [Pure]> {
  let summary = "Offset a tla.ptr by a scalar element count";
  let description = [{
    Returns a ``!tla.ptr`` advanced by ``offset`` **elements** of the pointer's pointee
    type (not bytes). The result preserves the source pointee type and address space;
    alignment is refined conservatively from a known constant offset (gcd), otherwise
    dropped to 1. The offset operand is an integer-or-index scalar SSA value.
  }];
  let arguments = (ins Tla_PtrType:$ptr, Tla_IntegerOrIndexType:$offset);
  let results = (outs Tla_PtrType:$result);
  let assemblyFormat = [{
    $ptr `,` $offset attr-dict `:` qualified(type($ptr)) `,` type($offset) `->` qualified(type($result))
  }];
  let hasVerifier = 1;
}

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

def Tla_CopyL0C2DstParamsOp : Tla_Op<"CopyL0C2DstParams", [Pure]> {
  let summary = "CopyL0C2DstParams op";
  let arguments = (ins
  // attributes
    I64Attr:$unit_flag,
    BoolAttr:$relu_enable,
    Tla_QuantModeAttr:$quant_mode,
    Tla_L0C2UBModeAttr:$l0c2ub_mode,
  // operands
    Optional<AnyTypeOf<[F32, Tla_TensorType]>>:$quant_scale_or_tensor
  );
  let results = (outs Tla_CopyL0C2DstParamsType:$params);
}

def Tla_CopyOp : Tla_Op<"copy", []> {
  let summary = "Copy";
  let description = [{
    Copy between tensors/views in different address spaces. Cube data-path
    copies (GM->L1, L1->L0A, L1->L0B, L0C->GM, L0C->UB, L0C->L1, L1->UB) must be nested
    inside a ``tla.cube`` region; vector staging copies (GM->UB, UB->GM, UB->L1)
    must be nested inside a ``tla.vector`` region.

    The atomic operation is enabled if ``atomic_mode`` provided, and currently only ``AtomicMode.ADD`` is supported, which is lowered to HIVM ``SetAtomicOp``

    A microscaling (MX) L1->L0A/L0B load is ``tla.copy_mx``, which carries the
    operand's e8m0 scale block. The frontend spells that as ``tla.copy`` with a
    ``scale=`` argument, mirroring AscendC's five-argument ``LoadData``, but it
    lowers to the separate op.
  }];
  let arguments = (ins
    Tla_TensorType:$dst, Tla_TensorType:$src,
    Optional<AnyTypeOf<[Tla_CopyL0C2DstParamsType]>>:$params,
    OptionalAttr<Tla_AtomicModeAttr>:$atomic_mode);
  let assemblyFormat = [{
    $dst `,` $src (`,` $params^)? attr-dict `:` qualified(type($dst)) `,` qualified(type($src)) (`,` qualified(type($params))^)?
  }];
  let hasVerifier = 1;
}

def Tla_CopyMxOp : Tla_Op<"copy_mx", []> {
  let summary = "Microscaling (MX) L1->L0 copy";
  let description = [{
    Copy an MX operand tile from L1 to L0A/L0B, attaching its shared-exponent
    scale block. Must be nested inside a ``tla.cube`` region.

    A separate op from ``tla.copy`` so that the plain copy keeps a fixed operand
    shape. The frontend still spells this as ``tla.copy(dst, src, scale=...)``,
    mirroring AscendC, where the MX load is the five-argument overload of
    ``LoadData`` rather than a distinct entry point.

    On this hardware the scale is consumed by the *load*, not by the matmul:
    ``tla.mmad_mx`` takes no scale operand, and the MX association is
    established when the operand lands in L0.

    ``scale`` is an L1 tile of OCP ``e8m0`` shared exponents, one per
    ``MX_SCALE_GROUP_NUM`` (32) elements along K. Its element type is the
    dialect's own ``!tla.f8e8m0`` (``tla.Float8E8M0`` on the Python side); like
    the other cube element formats it carries no LLVM lowering, so at the
    memref boundary it is buffered as i8, one byte per exponent. It must be
    tagged ``zZMxScale`` for an L1A and ``nNMxScale`` for an L1B.

    The destination L0 tile is what makes the matmul an MX matmul, so it must be
    consumed by ``tla.mmad_mx`` rather than ``tla.mmad``.
  }];
  let arguments = (ins Tla_TensorType:$dst, Tla_TensorType:$src, Tla_TensorType:$scale);
  let assemblyFormat = [{
    $dst `,` $src `,` $scale attr-dict `:`
    qualified(type($dst)) `,` qualified(type($src)) `,` qualified(type($scale))
  }];
  let hasVerifier = 1;
}

def Tla_FillOp : Tla_Op<"fill", []> {
  let summary = "Fill a region of an L1 zN/nZ tile with a scalar bit pattern";
  let description = [{
    Fill ``dst`` with ``value`` over the region ``[coord, coord + shape)``.
    Cube only: the destination must be an L1 tile tagged ``zN`` or ``nZ``, with
    a fp4/fp8 element type, and ``value`` must be 0 -- the op exists to
    zero the K-pad tail of an MX operand tile, nothing else.

    ``value`` travels as its i32 bit pattern (a uint16 pattern for the
    fp4/fp8 storage). The frontend crops ``shape`` against the tile's origin
    shape and the given coord before this op is formed. The device helper
    aligns the coordinate up to the C0 boundary (32 bytes) along the layout's
    C0 axis before filling, so a fill that starts or ends inside a C0 block
    degrades to the enclosing aligned blocks and a zero-length fill is a
    no-op.
  }];
  let arguments = (ins Tla_TensorType:$dst, I32:$value,
                         Tla_CoordType:$coord, Tla_ShapeType:$shape);
  let assemblyFormat = [{
    $dst `,` $value `,` $coord `,` $shape attr-dict `:`
    qualified(type($dst)) `,` type($value) `,` type($coord) `,` type($shape)
  }];
  let hasVerifier = 1;
}

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

def Tla_CubeOp : Tla_Op<"cube", [NoTerminator, SingleBlock]> {
  let summary = "Cube region";
  let description = [{
    Launch a cube region for matrix multiply and related operations.
  }];
  let regions = (region AnyRegion:$body);
  let assemblyFormat = [{
    $body attr-dict
  }];
}

def Tla_FlagOp : Tla_Op<"flag", [Pure]> {
  let summary = "Flag literal";
  let description = [{
    Materialize a static synchronization flag.
  }];
  let arguments = (ins StrAttr:$name, Tla_PipeAttr:$src_pipe, Tla_PipeAttr:$dst_pipe);
  let results = (outs Tla_FlagType:$flag);
  let assemblyFormat = [{
    $name attr-dict `->` qualified(type($flag))
  }];
}

def Tla_FuncOp : Tla_Op<"func", []> {
  let summary = "Tla kernel function";
  let description = [{
    Defines a Tla kernel entry point. The body is an isolated region with Tla IR operations.
  }];
  let arguments = (ins SymbolNameAttr:$sym_name, TypeAttr:$function_type);
  let regions = (region AnyRegion:$body);
  let hasCustomAssemblyFormat = 1;
}

def Tla_LoadOp : Tla_Op<"load", []> {
  let summary = "Load";
  let description = [{
    Load a UB tensor tile into vector or mask SSA. Dispatch is by result type:

    * ``!tla.vector`` — data load. Optional ``load_dist`` / ``unaligned_ub_access``.
      Dual-destination ``dintlv_b32`` produces a second result (even/odd b32
      elements), lowering to ``ave.hir.vload <DINTLV_B32>``. Unaligned path
      lowers to ``ave.hir.vload`` with ``vldas``/``vldus`` via HIVMAVE.
      Optional compile-time ``block_stride`` (i32 attribute) enables strided
      block load (AscendC ``vsldb``): one instruction gathers 8 DataBlocks
      whose heads are ``block_stride`` DataBlocks (32B units) apart;
      ``block_stride == 0`` repeats the first DataBlock. ``repeat_stride``
      (POST_MODE_NORMAL) is a compile-time address pre-offset in 32B
      DataBlocks. Strided load is mutually exclusive with ``load_dist`` and
      the dual result.
    * ``!tla.mask`` — continuous-aligned MaskReg load (AscendC
      ``LoadAlign(MaskReg&)`` / ``DIST_NORM`` / ``plds``). Source UB element
      type must be a 1/2/4-byte scalar (int or float). ``N`` on
      ``!tla.mask<N>`` is ``256 / sizeof(elem)`` from the UB element type;
      the tile supplies the address. Lowers via ``ave.hir.vload <NORM>`` on
      an ``i1`` memref view to ``hivm.plds.b8``.

    Must be nested inside a ``tla.vec.func`` region.
  }];
  let arguments = (ins Tla_TensorType:$source,
                       OptionalAttr<I32Attr>:$block_stride,
                       OptionalAttr<I32Attr>:$repeat_stride,
                       OptionalAttr<Tla_LoadDistAttr>:$load_dist,
                       OptionalAttr<UnitAttr>:$unaligned_ub_access);
  let results = (outs AnyTypeOf<[Tla_VectorSSAType, Tla_MaskSSAType]>:$result,
                      Optional<Tla_VectorSSAType>:$result2);
  let assemblyFormat = [{
    $source attr-dict `:` type($source) `->` qualified(type($result))
    ( `,` qualified(type($result2))^)?
  }];
  let hasVerifier = 1;
}

// Scratch alloc uses a static ``size_bytes`` attribute (``i64``) on ``tla.alloc_ptr``.
def Tla_AllocPtrOp : Tla_Op<"alloc_ptr", [Pure]> {
  let summary = "Static on-chip scratch bytes";
  let description = [{
    Reserves one statically assigned ``size_bytes`` on-chip scratch slot. The slot
    lifetime is the whole kernel: executing an alloc nested in a region or loop
    yields the same address on every dynamic execution. Address space and element
    alignment appear only on ``!tla.ptr<pointee, addrspace, align>``.
    ``tla-lower-ptr`` replaces the result with its static ``i64`` byte address.
  }];
  let arguments = (ins I64Attr:$size_bytes);
  let results = (outs Tla_PtrType:$ptr);
  let assemblyFormat = [{
    `{` `size_bytes` `=` $size_bytes `}` attr-dict `->` qualified(type($ptr))
  }];
  let hasVerifier = 1;
}

def Tla_DynamicUbBaseOp : Tla_Op<"dynamic_ub_base", [Pure]> {
  let summary = "Base address of the launch-sized UB region";
  let description = [{
    Yields the address just above every static ``tla.alloc_ptr`` in UB -- the
    start of a region whose size is chosen at launch rather than compiled in.

    Reserves nothing. Static allocations occupy the bottom of UB, so the region
    always begins at their high-water mark: a compile-time constant even though
    its extent is not. ``tla-lower-ptr`` replaces the result with that address,
    rounded up to the pointer's alignment.

    How much of the region is backed by real UB is set per launch. Touching
    past it is not diagnosed and does not reliably fault; it corrupts whatever
    the unclaimed UB is being used for.
  }];
  let results = (outs Tla_PtrType:$ptr);
  let assemblyFormat = [{
    attr-dict `->` qualified(type($ptr))
  }];
  let hasVerifier = 1;
}

def Tla_MakeTensorLikeOp : Tla_Op<"make_tensor_like", []> {
  let summary = "Make tensor like";
  let description = [{
    Reconstruct a typed local tensor from a raw allocation pointer and a reference tensor.
  }];
  let arguments = (ins Tla_PtrType:$ptr, Tla_TensorType:$like, Tla_LayoutTagAttr:$layoutTag);
  let results = (outs Tla_TensorType:$tile);
  let assemblyFormat = [{
    $ptr `like` $like `layoutTag` `(` $layoutTag `)` attr-dict `:` qualified(type($ptr)) `,` qualified(type($like)) `->` qualified(type($tile))
  }];
}

def Tla_MakeTensorOp : Tla_Op<"make_tensor", [Pure]> {
  let summary = "Construct a tensor from an explicit pointer, layout, and coord";
  let description = [{
    Build a `!tla.tensor<layout, coord, ptr>` from explicit operands. Unlike
    `tla.make_tensor_like` (which clones layout/coord from a reference tensor), this op
    takes the layout and coord directly as operands, mirroring the tensor type's
    `(layout, coord, ptr)` structure. Address space and element type come from the `ptr`
    operand's `!tla.ptr` type; the layout tag, shape, stride, and origin come from the
    `!tla.layout` operand.
  }];
  let arguments = (ins Tla_PtrType:$ptr, Tla_LayoutType:$layout, Tla_CoordType:$coord);
  let results = (outs Tla_TensorType:$result);
  let assemblyFormat = [{
    $ptr `,` $layout `,` $coord attr-dict `:`
    qualified(type($ptr)) `,` qualified(type($layout)) `,` qualified(type($coord)) `->` qualified(type($result))
  }];
}

def Tla_TensorDescOp : Tla_Op<"tensor_desc", [Pure]> {
  let summary = "A tile's fully-resolved tensor descriptor as a single op";
  let description = [{
    Materialized by ``tla-lower-tensor-desc`` from a tensor-view producer chain.
    ``base`` is the resolved root storage: either a builtin ``memref`` kernel
    argument or a ``!tla.ptr`` whose pointer transforms have already been applied.
    The remaining operands are the descriptor's SSA-valued four-slot layout shape
    and stride, two-dimensional origin shape, and two-dimensional coordinate.
    Linear layouts use ``[M, N, 1, 1]`` shape and ``[sM, sN, 1, 1]`` stride;
    NZFamily layouts preserve their four shape/stride leaves.

    Downstream passes consume this op without interpreting the producer chain.
  }];
  let arguments = (ins
    AnyTypeOf<[Tla_PtrType, AnyMemRef]>:$base,
    Index:$shape0, Index:$shape1, Index:$shape2, Index:$shape3,
    Index:$stride0, Index:$stride1, Index:$stride2, Index:$stride3,
    Index:$origin_shape0, Index:$origin_shape1,
    Index:$coord0, Index:$coord1);
  let results = (outs Tla_TensorType:$result);
  let assemblyFormat = [{
    $base `shape` `[` $shape0 `,` $shape1 `,` $shape2 `,` $shape3 `]`
    `stride` `[` $stride0 `,` $stride1 `,` $stride2 `,` $stride3 `]`
    `origin_shape` `[` $origin_shape0 `,` $origin_shape1 `]`
    `coord` `[` $coord0 `,` $coord1 `]`
    attr-dict `:` qualified(type($base)) `->` qualified(type($result))
  }];
  let hasVerifier = 1;
}

def Tla_MmadOp : Tla_Op<"mmad", []> {
  let summary = "Matrix multiply-accumulate";
  let description = [{
    Perform a matrix multiply-accumulate on tensor views. Must be nested inside a
    ``tla.cube`` region.

    ``compute_order`` is always supplied by the frontend. The lowering emits
    ``hivm.set_ctrl`` on ``CTRL[51]`` (0 = ``M_FIRST``, 1 = ``N_FIRST``): when
    every mmad in a function agrees on the value it is emitted once at the
    function entry; if a function mixes values it is emitted ahead of each mmad
    runtime call instead.

    ``hf32_mode`` is always supplied by the frontend. The lowering emits
    ``hivm.set_ctrl`` on ``CTRL[46]`` (enable HF32 rounding) and ``CTRL[47]``
    (rounding mode, ``HF32_NEAREST_ZERO`` -> 1 / ``HF32_NEAREST_EVEN`` -> 0,
    mirroring ``AscendC::SetHF32Mode`` / ``AscendC::SetHF32TransMode``): when
    every mmad in a function agrees on the value it is emitted once at the
    function entry; if a function mixes values it is emitted ahead of each mmad
    runtime call instead.
  }];
  let arguments = (ins Tla_TensorType:$acc, Tla_TensorType:$lhs, Tla_TensorType:$rhs,
                       I1:$init_c, I64:$unit_flag,
                       Tla_ComputeOrderAttr:$compute_order,
                       Tla_HF32ModeAttr:$hf32_mode);
  let assemblyFormat = [{
    $acc `,` $lhs `,` $rhs `,` $init_c `,` $unit_flag attr-dict `:`
    qualified(type($acc)) `,` qualified(type($lhs)) `,` qualified(type($rhs)) `,` type($init_c) `,` type($unit_flag)
  }];
  let hasVerifier = 1;
}

def Tla_MmadMxOp : Tla_Op<"mmad_mx", []> {
  let summary = "Microscaling (MX) matrix multiply-accumulate";
  let description = [{
    Perform a microscaling matrix multiply-accumulate on tensor views. Must be
    nested inside a ``tla.cube`` region.

    A distinct op rather than a flag on ``tla.mmad``, mirroring the hardware:
    the cube has a dedicated ``mad_mx`` instruction, which AscendC exposes as
    ``asc_mmad_mx``.

    Like that instruction, this op takes **no scale operand**. ``lhs``/``rhs``
    must have been loaded by ``tla.copy`` with its ``scale`` operand, which is
    what attaches the e8m0 block to the L0 tile; the scale then lives in a side
    buffer addressed from the L0 tile rather than being passed here.

    A packed-fp4 operand names its encoding in its own element type
    (``!tla.f4e2m1`` / ``!tla.f4e1m2``), so there is nothing to state here and
    nothing that can disagree with the load that wrote the tile.
  }];
  let arguments = (ins Tla_TensorType:$acc, Tla_TensorType:$lhs, Tla_TensorType:$rhs,
                       I1:$init_c, I64:$unit_flag,
                       Tla_ComputeOrderAttr:$compute_order);
  let assemblyFormat = [{
    $acc `,` $lhs `,` $rhs `,` $init_c `,` $unit_flag attr-dict `:`
    qualified(type($acc)) `,` qualified(type($lhs)) `,` qualified(type($rhs)) `,` type($init_c) `,` type($unit_flag)
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

def Tla_DebugPrintOp : Tla_Op<"debug_print", [MemoryEffects<[MemWrite]>]> {
  let summary = "Typed frontend debug print op";
  let description = [{
    Side-effecting TLA IR for legacy one-scalar debug print or compile-time
    string/formatted scalar print. The legacy assembly remains:

      tla.debug_print %value : type

    Formatted calls carry the validated user format string as metadata plus
    zero or more runtime scalar operands:

      tla.debug_print format "x={}"(%value) : (type)

    Formatted calls lower through the same private debug-print backend and
    scalar FIFO transport as legacy scalar debug print.

    Legacy operands support f16, f32, i8, i16, i32, u8, u16, and u32.
    Formatted operands support signless i32 and f32.
  }];
  let arguments = (ins Variadic<AnyType>:$values,
                       OptionalAttr<StrAttr>:$format);
  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
}

def Tla_CallExternOp : Tla_Op<"call_extern", [MemoryEffects<[MemRead, MemWrite]>]> {
  let summary = "Call a user-provided device function";
  let description = [{
    Calls one C ABI device function supplied by user LLVM bitcode. Pointer
    operands remain !tla.ptr until tla-lower-ptr converts them to i64 byte
    addresses. v1 has no results and conservatively models unknown memory
    reads and writes.
  }];
  let arguments = (ins FlatSymbolRefAttr:$callee, Variadic<AnyType>:$operands);
  let assemblyFormat = [{
    $callee `(` $operands `)` attr-dict `:` functional-type($operands, results)
  }];
  let hasVerifier = 1;
}

def Tla_PrintTensorOp : Tla_Op<"print_tensor", [MemoryEffects<[MemWrite]>]> {
  let summary = "Print a supported tensor prefix from GM, UB, L1, or L0C";
  let description = [{
    Side-effecting native tensor print marker. The Python API accepts rank-1 or
    rank-2 tensors with element type f16, f32, i8, i16, i32, u8, u16, or u32.
    GM is supported in AIC and AIV regions, UB in AIV regions, and the supported
    zN/nZ L1 and L0C layouts in AIC regions. Values are read as a contiguous
    physical prefix from the effective tensor address; ``shape`` is logical
    display metadata, not a request to gather through tensor strides. Dynamic
    shape leaves use ``-1`` and are materialized from the descriptor during
    lowering. ``length`` may be a runtime i64. ``dynamic_execution`` records
    that the site is nested under runtime control flow so the host warns rather
    than rejecting an aggregate FIFO bound that cannot be known statically.
  }];
  let arguments = (ins Tla_TensorType:$value, I64:$length,
                       DenseI64ArrayAttr:$shape,
                       OptionalAttr<UnitAttr>:$dynamic_execution);
  let assemblyFormat = [{
    $value `length` `=` $length `shape` `=` $shape attr-dict `:`
    type($value) `,` type($length)
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

def Tla_MakeShapeOp : Tla_Op<"make_shape", [Pure]> {
  let summary = "Shape value";
  let description = [{
    Build a ``!tla.shape`` from operands for each dynamic dimension; rank and static sizes are fixed by the result type.
  }];
  let arguments = (ins Variadic<Index>:$dyn_elems);
  let results = (outs Tla_ShapeType:$result);
  let hasCustomAssemblyFormat = 1;
}

def Tla_MakeCoordOp : Tla_Op<"make_coord", [Pure]> {
  let summary = "Coord value";
  let description = [{
    Build a ``!tla.coord`` from operands for each dynamic leaf; structure and static leaves are fixed by the result type.
  }];
  let arguments = (ins Variadic<Index>:$dyn_elems);
  let results = (outs Tla_CoordType:$result);
  let hasCustomAssemblyFormat = 1;
}

def Tla_MakeStrideOp : Tla_Op<"make_stride", [Pure]> {
  let summary = "Stride value";
  let description = [{
    Build a ``!tla.stride`` from operands for each dynamic leaf; structure and static leaves are fixed by the result type.
  }];
  let arguments = (ins Variadic<Index>:$dyn_elems);
  let results = (outs Tla_StrideType:$result);
  let hasCustomAssemblyFormat = 1;
}

def Tla_MakeLayoutOp : Tla_Op<"make_layout", [Pure]> {
  let summary = "Layout value";
  let description = [{
    Combine a packed ``!tla.shape`` and ``!tla.stride`` into a ``!tla.layout``, analogous to ``tla.make_layout(shape, stride)``.
    The result type encodes tuple-tree structure as ``!tla.layout<!tla.shape<shape_repr>, !tla.stride<stride_repr>, layout_tag>``, or
    ``!tla.layout<!tla.shape<shape_repr>, !tla.stride<stride_repr>, !tla.shape<origin_shape_repr>, layout_tag>`` when an explicit origin shape is used.

    Optional ``origin_shape`` records the logical extent (e.g. pre-padding matrix size). When omitted,
    consumers should treat the layout's logical origin shape as identical to ``shape``.
  }];
  let arguments = (ins Tla_ShapeType:$shape,
                       Tla_StrideType:$stride,
                       Optional<Tla_ShapeType>:$origin_shape,
                       OptionalAttr<Tla_LayoutTagAttr>:$layoutTag);
  let results = (outs Tla_LayoutType:$result);
  let assemblyFormat = [{
    $shape `,` $stride (`origin` $origin_shape^)? attr-dict `:`
    qualified(type($shape)) `,` qualified(type($stride))
    (`origin` qualified(type($origin_shape))^)? `->` qualified(type($result))
  }];
}

def Tla_ReturnOp : Tla_Op<"return", [Terminator]> {
  let summary = "Return";
  let description = [{
    Terminator for tla.func regions.
  }];
  let assemblyFormat = [{
    attr-dict
  }];
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

def Tla_StoreOp : Tla_Op<"store", [AttrSizedOperandSegments]> {
  let summary = "Store";
  let description = [{
    Store vector or mask SSA into a UB tensor tile. Dispatch is by source type:

    * ``!tla.vector`` — data store. Optional predicate ``mask`` controls which
      lanes are written. Optional ``unaligned_ub_access`` lowers to unaligned
      UB store (``vstas``/``vstus`` via HIVMAVE).
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
  let assemblyFormat = [{
    $dest `,` $source (`mask` $mask^)? (`stride` $block_stride^)? attr-dict `:`
    type($dest) `,` qualified(type($source)) (`mask` qualified(type($mask))^)?
    (`block_stride` qualified(type($block_stride))^)?
  }];
  let hasVerifier = 1;
}

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

def Tla_TileViewOp : Tla_Op<"tile_view", []> {
  let summary = "Tile view";
  let description = [{
    Create a tensor view over an existing ``!tla.tensor`` using packed
    ``!tla.shape`` and ``!tla.coord`` values.
  }];
  let arguments = (ins Tla_TensorType:$source, Tla_ShapeType:$shape, Tla_CoordType:$coord, OptionalAttr<Tla_LayoutTagAttr>:$layoutTag);
  let results = (outs Tla_TensorType:$view);
  let assemblyFormat = [{
    $source `,` $shape `,` $coord attr-dict `:` qualified(type($source)) `,` qualified(type($shape)) `,` qualified(type($coord)) `->` qualified(type($view))
  }];
}

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

def Tla_SyncThreadsOp : Tla_Op<"arch.sync_threads", []> {
  let summary = "SIMT block-wide thread barrier";
  let description = [{
    Synchronize every thread of the enclosing ``tla.vec.func`` with
    ``mode = "simt"``. Required between a write and a read of shared data by
    different threads. Lowers to ``hivm_regbaseintrins.sync_threads``.
  }];
  let assemblyFormat = [{ attr-dict }];
}

def Tla_SimtLoadOp : Tla_Op<"simt_load", [MemoryEffects<[MemRead]>]> {
  let summary = "Per-thread tensor element read";
  let description = [{
    Read one element of a ``!tla.tensor`` from inside a ``tla.vec.func`` with
    ``mode = "simt"``: one thread, one element. The SIMT counterpart of
    ``tla.scalar_load``, which stays the general descriptor-based access.
    ``tla-vector-region`` lowers this to ``memref.load`` on the outlined vector
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
    ``tla-vector-region`` to ``memref.store`` on the outlined vector function's
    parameter.
  }];
  let arguments = (ins Tla_TensorType:$dest, Variadic<Index>:$indices,
                       Tla_ScalarValueType:$value);
  let assemblyFormat = [{
    $dest `[` $indices `]` `,` $value attr-dict `:` type($dest) `,` type($value)
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
  let summary = "Vector gather on VectorSSA values";
  let description = [{
    Gathers elements from ``x`` to ``result``according to the indices in ``y``.
  }];
  let arguments = (ins Tla_TensorType:$x, Tla_VectorSSAType:$y,
                       Optional<Tla_MaskSSAType>:$mask);
  let results = (outs Tla_VectorSSAType:$result);
  let assemblyFormat = [{
    $x `,` $y (`mask` $mask^)? attr-dict `:`
    type($x) `,` qualified(type($y)) (`mask` qualified(type($mask))^)? `->` qualified(type($result))
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

def Tla_WhereOp : Tla_Op<"where", [Pure]> {
  let summary = "Vector select on VectorSSA values";
  let description = [{
    Element-wise select on VectorSSA values. For each lane the
    result holds the corresponding lane of ``x`` where the mask is
    active and ``y`` elsewhere. ``x`` and ``y`` must share the same VectorSSA type.
  }];
  let arguments = (ins Tla_MaskSSAType:$mask, Tla_VectorSSAType:$x, Tla_VectorSSAType:$y);
  let results = (outs Tla_VectorSSAType:$result);
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

def Tla_VectorOp : Tla_Op<"vector", [NoTerminator, SingleBlock]> {
  let summary = "Vector region";
  let description = [{
    Launch a vector region for post-processing and epilogue ops.
  }];
  let regions = (region AnyRegion:$body);
}

def Tla_VecFuncOp : Tla_Op<"vec.func", [NoTerminator, SingleBlock]> {
  let summary = "Vector compute function region";
  let description = [{
    Group register-resident vector compute ops inside a vector region. Must be
    nested inside a ``tla.vector`` region.

    ``mode`` selects the execution model: ``"simd"`` (the default when the
    attribute is absent) runs the body as whole-vector ops, ``"simt"`` runs it
    once per thread.

    ``thread_block_dim`` gives the ``(x, y, z)`` thread extents of one block and
    is valid only with ``mode = "simt"``, where it is required. Each extent must
    be positive and the product may not exceed 2048 threads per block, which is
    both the supported maximum and what fits the ``simt_entry`` attribute the
    lowering builds from it.
  }];
  let regions = (region AnyRegion:$body);
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
