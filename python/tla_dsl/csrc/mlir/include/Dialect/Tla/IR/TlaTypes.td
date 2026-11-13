//===----------------------------------------------------------------------===//
// Type definitions and type constraints of the Tla dialect.
//
// Included by Dialect/Tla/IR/Tla.td, the file mlir-tblgen is run on.
//===----------------------------------------------------------------------===//

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

// UnAlign align-register carrier threaded as ordinary SSA across UPDATE ops and
// scf.for iter_args. Not a developer-facing tile API.
def Tla_UregType : TypeDef<TlaDialect, "Ureg", []> {
  let mnemonic = "ureg";
  let summary = "UnAlign ureg stream carrier (compiler-internal SSA).";
}

def Tla_ScalarValueType : AnyTypeOf<[AnyInteger, AnyFloat, Index],
    "TLA scalar SSA value type">;

def Tla_IntegerOrIndexType : AnyTypeOf<[AnyInteger, Index],
    "TLA integer-or-index SSA value type">;

// Element encodings that this MLIR has no builtin type for.
//
// LLVM 19 ships Float8E4M3FN / Float8E5M2 and nothing narrower: Float4E2M1FN and
// Float8E8M0FNU both arrive in LLVM 20. Spelling them as dialect types rather
// than borrowing i4 / i8 keeps the *format* in the type, where it belongs --
// e2m1 and e1m2 are the same width and differ only in exponent/mantissa split,
// so a shared integer carrier cannot tell them apart and the encoding has to be
// smuggled alongside on an attribute that nothing verifies against the type.
//
// Logical FP4 and E8M0 describe Cube storage. HiF8 also supports explicit
// SIMD register conversion and transport; arithmetic requires conversion.
// At the memref boundary these are buffered as i8 -- see storageTypeAndCount.
//
// With LLVM 20 the logical FP4 and E8M0 types can use the corresponding
// builtin formats. HiF8 remains a separate encoding.
class Tla_EncodedElementType<string name, string typeMnemonic, string desc>
    : TypeDef<TlaDialect, name, []> {
  let mnemonic = typeMnemonic;
  let summary = desc;
}

def Tla_Float4E2M1Type : Tla_EncodedElementType<"Float4E2M1", "f4e2m1",
    "Packed 4-bit float, 2-bit exponent / 1-bit mantissa (OCP MXFP4).">;
def Tla_Float4E1M2Type : Tla_EncodedElementType<"Float4E1M2", "f4e1m2",
    "Packed 4-bit float, 1-bit exponent / 2-bit mantissa.">;
def Tla_Float8E8M0Type : Tla_EncodedElementType<"Float8E8M0", "f8e8m0",
    "8-bit shared exponent (OCP microscaling scale block).">;

// Byte-addressable numeric encoding, carried as i8 at storage boundaries.
def Tla_HiFloat8Type : Tla_EncodedElementType<"HiFloat8", "hif8",
    "HiFloat8 payload with tapered exponent and mantissa fields.">;


def Tla_Float4E2M1x2Type : TypeDef<TlaDialect, "Float4E2M1x2", []> {
  let mnemonic = "f4e2m1x2";
  let summary = "One byte holding two E2M1 values, low nibble first.";
}

def Tla_Float4E1M2x2Type : TypeDef<TlaDialect, "Float4E1M2x2", []> {
  let mnemonic = "f4e1m2x2";
  let summary = "One byte holding two E1M2 values, low nibble first.";
}

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

def Tla_CopyL0C2DstParamsType : TypeDef<TlaDialect, "CopyL0C2DstParams", []> {
  let mnemonic = "copy_l0c2dst_params";
  let summary = "CopyL0C2DstParams for tla.copy.";
  let assemblyFormat = "";
}
