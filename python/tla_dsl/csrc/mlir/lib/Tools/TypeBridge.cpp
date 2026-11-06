#include "Dialect/Tla/IR/TlaAttrs.h"
#include "Dialect/Tla/IR/TlaDialect.h"
#include "Dialect/Tla/IR/TlaOps.h"
#include "Dialect/Tla/IR/TlaTypes.h"
#include "Passes.h"
#include "Passes/TlaSimtUbLimits.h"
#include "Passes/TlaTensorDescriptor.h"
#include "Tools/AddressSpaceConversion.h"
#include "Tools/CompilePipeline.h"

#include "mlir/Bindings/Python/PybindAdaptors.h"
#include "mlir/CAPI/IR.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "pybind11/stl.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/raw_ostream.h"

#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/HACC/IR/HACC.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

using namespace mlir;

namespace py = pybind11;

namespace {

InFlightDiagnostic emitBridgeError(MLIRContext* ctx, StringRef message)
{
    return emitError(UnknownLoc::get(ctx), message);
}

StringRef diagnosticSeverityText(DiagnosticSeverity severity)
{
    switch (severity) {
        case DiagnosticSeverity::Error:
            return "error";
        case DiagnosticSeverity::Warning:
            return "warning";
        case DiagnosticSeverity::Note:
            return "note";
        case DiagnosticSeverity::Remark:
            return "remark";
    }
    llvm_unreachable("unknown MLIR diagnostic severity");
}

struct BridgeSourceLocation {
    std::string filename;
    unsigned line;
    unsigned column;
};

struct BridgeDiagnostic {
    std::string severity;
    std::string message;
    std::vector<BridgeSourceLocation> locations;
    std::string rendered;
};

std::vector<BridgeSourceLocation> diagnosticLocations(Location location)
{
    std::vector<BridgeSourceLocation> locations;
    auto append = [&](FileLineColLoc fileLocation) {
        BridgeSourceLocation candidate{
            fileLocation.getFilename().str(), fileLocation.getLine(), fileLocation.getColumn()};
        if (!llvm::any_of(locations, [&](const BridgeSourceLocation& existing) {
                return existing.filename == candidate.filename && existing.line == candidate.line &&
                       existing.column == candidate.column;
            }))
            locations.push_back(std::move(candidate));
    };
    // Preserve every nested source coordinate in MLIR's deterministic
    // traversal order. This covers fused, named, call-site, and opaque
    // locations without requiring the bridge to interpret their storage.
    location->walk([&](Location nested) {
        if (auto fileLocation = dyn_cast<FileLineColLoc>(nested))
            append(fileLocation);
        return WalkResult::advance();
    });

    return locations;
}

BridgeDiagnostic captureDiagnostic(Diagnostic& diagnostic)
{
    BridgeDiagnostic result;
    result.severity = diagnosticSeverityText(diagnostic.getSeverity()).str();
    result.message = diagnostic.str();
    result.locations = diagnosticLocations(diagnostic.getLocation());
    llvm::raw_string_ostream stream(result.rendered);
    if (!isa<UnknownLoc>(diagnostic.getLocation())) {
        diagnostic.getLocation().print(stream);
        stream << ": ";
    }
    stream << result.severity << ": " << result.message;
    stream.flush();
    return result;
}

void captureDiagnosticRecords(Diagnostic& diagnostic, std::vector<BridgeDiagnostic>& records)
{
    records.push_back(captureDiagnostic(diagnostic));
    // The diagnostic engine invokes handlers for the primary diagnostic only.
    // Preserve attached notes immediately after it, matching MLIR's output
    // order without adding a second, nested Python diagnostic schema.
    for (Diagnostic& note : diagnostic.getNotes())
        captureDiagnosticRecords(note, records);
}

py::dict diagnosticToPython(const BridgeDiagnostic& diagnostic)
{
    py::dict result;
    result["severity"] = diagnostic.severity;
    result["message"] = diagnostic.message;
    py::list locations;
    for (const BridgeSourceLocation& candidate : diagnostic.locations) {
        py::dict item;
        item["filename"] = candidate.filename;
        item["line"] = candidate.line;
        item["column"] = candidate.column;
        locations.append(std::move(item));
    }
    result["locations"] = std::move(locations);
    result["rendered"] = diagnostic.rendered;
    return result;
}

MLIRContext* bridgeContext(MlirContext context)
{
    MLIRContext* ctx = unwrap(context);
    if (!ctx)
        throw py::value_error("expected a non-null mlir.ir.Context");
    ctx->getOrLoadDialect<::tla::TlaDialect>();
    // Bridged GM memref types carry #hivm.address_space; Attribute storage requires
    // HIVMDialect to be loaded before AddressSpaceAttr::get.
    ctx->getOrLoadDialect<hivm::HIVMDialect>();
    return ctx;
}

void loadTlaDialect(MlirContext context)
{
    (void)bridgeContext(context);
}

ModuleOp moduleFromCapsule(MlirModule cModule)
{
    if (mlirModuleIsNull(cModule))
        throw py::type_error("Expected an mlir.ir.Module object.");
    ModuleOp module = unwrap(cModule);
    if (!module)
        throw py::value_error("Failed to unwrap mlir.ir.Module capsule.");
    return module;
}

bool containsPassName(const std::vector<std::string>& names, Pass* pass)
{
    StringRef argument = pass->getArgument();
    return llvm::any_of(names, [&](const std::string& name) { return argument == name; });
}

Type bridgeType(MlirType type, StringRef description = "mlir.ir.Type")
{
    Type unwrapped = unwrap(type);
    if (!unwrapped)
        throw py::type_error(std::string("expected a non-null ") + description.str());
    return unwrapped;
}

MlirType toMlirType(Type type, StringRef description)
{
    if (!type)
        throw py::value_error(std::string("failed to construct ") + description.str());
    return wrap(type);
}

ArrayRef<int64_t> treeRef(const std::vector<int64_t>& tree, StringRef description)
{
    if (tree.empty())
        throw py::value_error(description.str() + " must be non-empty");
    return ArrayRef<int64_t>(tree.data(), tree.size());
}

::std::optional<::AddressSpace> parseRequiredAddressSpace(const std::string& addrspaceText)
{
    auto addrspace = ::symbolizeAddressSpace(addrspaceText);
    if (!addrspace)
        throw py::value_error("unknown Tla address space: " + addrspaceText);
    return addrspace;
}

::std::optional<::LayoutTag> parseRequiredLayoutTag(const std::string& layoutText)
{
    auto layout = ::symbolizeLayoutTag(layoutText);
    if (!layout)
        throw py::value_error("unknown Tla layout tag: " + layoutText);
    return layout;
}

::tla::ShapeType makeShape(MLIRContext* ctx, const std::vector<int64_t>& tree, StringRef diagnostic)
{
    return ::tla::ShapeType::getChecked(
        [&] { return emitBridgeError(ctx, diagnostic); }, ctx, treeRef(tree, "tla.shape"));
}

::tla::StrideType makeStride(MLIRContext* ctx, const std::vector<int64_t>& tree, StringRef diagnostic)
{
    return ::tla::StrideType::getChecked(
        [&] { return emitBridgeError(ctx, diagnostic); }, ctx, treeRef(tree, "tla.stride"));
}

::tla::CoordType makeCoord(MLIRContext* ctx, const std::vector<int64_t>& tree, StringRef diagnostic)
{
    return ::tla::CoordType::getChecked(
        [&] { return emitBridgeError(ctx, diagnostic); }, ctx, treeRef(tree, "tla.coord"));
}

MlirType shapeTypeGet(MlirContext context, const std::vector<int64_t>& tree)
{
    MLIRContext* ctx = bridgeContext(context);
    return toMlirType(makeShape(ctx, tree, "invalid tla.shape type bridge input"), "tla.shape");
}

MlirType coordTypeGet(MlirContext context, const std::vector<int64_t>& tree)
{
    MLIRContext* ctx = bridgeContext(context);
    return toMlirType(makeCoord(ctx, tree, "invalid tla.coord type bridge input"), "tla.coord");
}

MlirType strideTypeGet(MlirContext context, const std::vector<int64_t>& tree)
{
    MLIRContext* ctx = bridgeContext(context);
    return toMlirType(makeStride(ctx, tree, "invalid tla.stride type bridge input"), "tla.stride");
}

template <typename TlaTypeT>
bool typeIs(MlirType type)
{
    return static_cast<bool>(dyn_cast_or_null<TlaTypeT>(bridgeType(type)));
}

template <typename TlaTypeT>
TlaTypeT checkedTlaType(MlirType type, StringRef description)
{
    auto tlaType = dyn_cast_or_null<TlaTypeT>(bridgeType(type));
    if (!tlaType)
        throw py::type_error(std::string("expected ") + description.str());
    return tlaType;
}

MlirType ptrTypeGet(MlirContext context, MlirType pointeeType, const std::string& addrspaceText, unsigned alignment)
{
    MLIRContext* ctx = bridgeContext(context);
    Type pointee = bridgeType(pointeeType, "pointee type");
    auto addrspace = parseRequiredAddressSpace(addrspaceText);
    return toMlirType(::tla::PtrType::get(ctx, pointee, *addrspace, alignment), "tla.ptr");
}

MlirType ptrPointeeTypeGet(MlirType ptrType)
{
    auto ptr = checkedTlaType<::tla::PtrType>(ptrType, "!tla.ptr type");
    return toMlirType(ptr.getPointee(), "ptr pointee type");
}

std::string ptrAddrspace(MlirType ptrType)
{
    auto ptr = checkedTlaType<::tla::PtrType>(ptrType, "!tla.ptr type");
    return stringifyAddressSpace(ptr.getAddrspace()).str();
}

unsigned ptrAlignment(MlirType ptrType)
{
    auto ptr = checkedTlaType<::tla::PtrType>(ptrType, "!tla.ptr type");
    return ptr.getAlignment();
}

MlirType tensorPtrTypeGet(MlirType tensorType)
{
    auto tensor = checkedTlaType<::tla::TlaTensorType>(tensorType, "!tla.tensor type");
    return toMlirType(tensor.getPtr(), "tla.tensor backing ptr type");
}

MlirType layoutTypeFromComponentsGet(
    MlirContext context, MlirType shapeType, MlirType strideType, py::object originShapeType,
    const std::string& layoutText)
{
    MLIRContext* ctx = bridgeContext(context);
    auto shape = checkedTlaType<::tla::ShapeType>(shapeType, "!tla.shape type");
    auto stride = checkedTlaType<::tla::StrideType>(strideType, "!tla.stride type");
    ::tla::ShapeType origin = shape;
    if (!originShapeType.is_none())
        origin = checkedTlaType<::tla::ShapeType>(originShapeType.cast<MlirType>(), "!tla.shape type");
    auto layout = parseRequiredLayoutTag(layoutText);
    Type type = ::tla::LayoutType::getChecked(
        [&] { return emitBridgeError(ctx, "invalid tla.layout component type bridge input"); }, ctx, shape, stride,
        origin, *layout);
    return toMlirType(type, "tla.layout");
}

MlirType layoutTypeGet(
    MlirContext context, const std::vector<int64_t>& shapeTree, const std::vector<int64_t>& strideTree,
    py::object originTreeObject, const std::string& layoutText)
{
    MLIRContext* ctx = bridgeContext(context);
    auto shape = makeShape(ctx, shapeTree, "invalid tla.layout shape bridge input");
    auto stride = makeStride(ctx, strideTree, "invalid tla.layout stride bridge input");
    ::tla::ShapeType origin = shape;
    if (!originTreeObject.is_none()) {
        auto originTree = originTreeObject.cast<std::vector<int64_t>>();
        origin = makeShape(ctx, originTree, "invalid tla.layout origin bridge input");
    }
    auto layout = parseRequiredLayoutTag(layoutText);
    Type type = ::tla::LayoutType::getChecked(
        [&] { return emitBridgeError(ctx, "invalid tla.layout type bridge input"); }, ctx, shape, stride, origin,
        *layout);
    return toMlirType(type, "tla.layout");
}

MlirType vectorSSAElementTypeGet(MlirType vectorType)
{
    auto vector = checkedTlaType<::tla::VectorSSAType>(vectorType, "!tla.vector type");
    return toMlirType(vector.getElementType(), "vector element type");
}

py::object vectorSSAValidLanesGet(MlirType vectorType)
{
    auto vector = checkedTlaType<::tla::VectorSSAType>(vectorType, "!tla.vector type");
    if (vector.getValidLanes() == ShapedType::kDynamic)
        return py::none();
    return py::int_(vector.getValidLanes());
}

MlirType vectorSSATypeGet(MlirContext context, py::object validLanes, MlirType elementType)
{
    MLIRContext* ctx = bridgeContext(context);
    int64_t lanes = validLanes.is_none() ? ShapedType::kDynamic : validLanes.cast<int64_t>();
    Type element = bridgeType(elementType, "element type");
    Type type = ::tla::VectorSSAType::getChecked(
        [&] { return emitBridgeError(ctx, "invalid tla.vector type bridge input"); }, ctx, lanes, element);
    return toMlirType(type, "tla.vector");
}

MlirType maskSSATypeGet(MlirContext context, int64_t physicalLanes)
{
    MLIRContext* ctx = bridgeContext(context);
    Type type = ::tla::MaskSSAType::getChecked(
        [&] { return emitBridgeError(ctx, "invalid tla.mask type bridge input"); }, ctx, physicalLanes);
    return toMlirType(type, "tla.mask");
}

int64_t maskSSAPhysicalLanesGet(MlirType maskType)
{
    return checkedTlaType<::tla::MaskSSAType>(maskType, "!tla.mask<N> type").getPhysicalLanes();
}

MlirType flagTypeGet(MlirContext context)
{
    MLIRContext* ctx = bridgeContext(context);
    return toMlirType(::tla::FlagType::get(ctx), "tla.flag");
}

MlirType crossFlagTypeGet(MlirContext context, int64_t mode)
{
    MLIRContext* ctx = bridgeContext(context);
    return toMlirType(::tla::CrossFlagType::get(ctx, mode), "tla.cross_flag");
}

int64_t crossFlagMode(MlirType type)
{
    return checkedTlaType<::tla::CrossFlagType>(type, "!tla.cross_flag type").getMode();
}

// The cube element formats this MLIR has no builtin for. See Tla.td.
MlirType float4E2M1TypeGet(MlirContext context)
{
    MLIRContext* ctx = bridgeContext(context);
    return toMlirType(::tla::Float4E2M1Type::get(ctx), "tla.f4e2m1");
}

MlirType float4E1M2TypeGet(MlirContext context)
{
    MLIRContext* ctx = bridgeContext(context);
    return toMlirType(::tla::Float4E1M2Type::get(ctx), "tla.f4e1m2");
}

MlirType float8E8M0TypeGet(MlirContext context)
{
    MLIRContext* ctx = bridgeContext(context);
    return toMlirType(::tla::Float8E8M0Type::get(ctx), "tla.f8e8m0");
}

MlirType mutexTypeGet(MlirContext context)
{
    MLIRContext* ctx = bridgeContext(context);
    return toMlirType(::tla::MutexType::get(ctx), "tla.mutex");
}

MlirType copyL0C2DstParamsTypeGet(MlirContext context)
{
    MLIRContext* ctx = bridgeContext(context);
    return toMlirType(::tla::CopyL0C2DstParamsType::get(ctx), "tla.CopyL0C2DstParams");
}

MlirType tensorTypeGet(
    MlirContext context, const std::vector<int64_t>& shapeTree, const std::vector<int64_t>& strideTree,
    const std::vector<int64_t>& coordTree, const std::vector<int64_t>& originShapeTree, MlirType elementType,
    const std::string& addrspaceText, const std::string& layoutText, unsigned ptrAlignment)
{
    MLIRContext* ctx = bridgeContext(context);
    auto shape = makeShape(ctx, shapeTree, "invalid tla.tensor shape bridge input");
    auto stride = makeStride(ctx, strideTree, "invalid tla.tensor stride bridge input");
    auto coord = makeCoord(ctx, coordTree, "invalid tla.tensor coord bridge input");
    auto originShape = makeShape(ctx, originShapeTree, "invalid tla.tensor origin bridge input");
    Type element = bridgeType(elementType, "element type");
    auto addrspace = parseRequiredAddressSpace(addrspaceText);
    auto layoutTag = parseRequiredLayoutTag(layoutText);
    auto layout = ::tla::LayoutType::getChecked(
        [&] { return emitBridgeError(ctx, "invalid tla.tensor layout bridge input"); }, ctx, shape, stride, originShape,
        *layoutTag);
    auto ptr = ::tla::PtrType::get(ctx, element, *addrspace, ptrAlignment);
    Type type = ::tla::TlaTensorType::getChecked(
        [&] { return emitBridgeError(ctx, "invalid tla.tensor type bridge input"); }, ctx, layout, coord, ptr);
    return toMlirType(type, "tla.tensor");
}

/// Unified dynamic GM ABI memref from ``!tla.tensor`` (schema-v4: 4 size/stride slots).
MlirType dynamicGmMemrefTypeGet(MlirType tensorType)
{
    Type tlaTensor = bridgeType(tensorType, "tla.tensor type");
    if (!tlaTensor.getContext())
        throw py::value_error("expected tla.tensor type with a live MLIRContext");
    (void)bridgeContext(wrap(tlaTensor.getContext()));
    FailureOr<MemRefType> bridged = ::tla::bridgeTlaTensorType(tlaTensor);
    if (failed(bridged))
        throw py::value_error("failed to bridge tla.tensor storage type to memref");
    Type elem = bridged->getElementType();
    MLIRContext* ctx = elem.getContext();
    Attribute gmSpace = hivm::AddressSpaceAttr::get(ctx, hivm::AddressSpace::GM);
    SmallVector<int64_t, 4> dynShape(4, ShapedType::kDynamic);
    SmallVector<int64_t, 4> dynStrides(4, ShapedType::kDynamic);
    auto layout = StridedLayoutAttr::get(ctx, ShapedType::kDynamic, dynStrides);
    Type memref = MemRefType::get(dynShape, elem, layout, gmSpace);
    return toMlirType(memref, "dynamic GM memref type");
}

std::optional<std::string> tlaTypeCategory(MlirType type)
{
    Type unwrapped = bridgeType(type);
    if (isa<::tla::TlaTensorType>(unwrapped))
        return "tensor";
    if (isa<::tla::VectorSSAType>(unwrapped))
        return "vector_ssa";
    if (isa<::tla::MaskSSAType>(unwrapped))
        return "mask_ssa";
    if (isa<::tla::ShapeType>(unwrapped))
        return "shape";
    if (isa<::tla::CoordType>(unwrapped))
        return "coord";
    if (isa<::tla::StrideType>(unwrapped))
        return "stride";
    if (isa<::tla::LayoutType>(unwrapped))
        return "layout";
    if (isa<::tla::PtrType>(unwrapped))
        return "pointer";
    if (isa<::tla::FlagType>(unwrapped))
        return "flag";
    if (isa<::tla::CrossFlagType>(unwrapped))
        return "cross_flag";
    if (isa<::tla::MutexType>(unwrapped))
        return "mutex";
    if (isa<::tla::CopyL0C2DstParamsType>(unwrapped))
        return "CopyL0C2DstParams";
    return std::nullopt;
}

// 0 is a scalar, 1 is a launchable GM pointer, and -1 is a pointer whose
// address space cannot participate in the host launch ABI.
using KernelPointerProvenance = llvm::StringMap<SmallVector<int8_t, 8>>;

KernelPointerProvenance collectKernelPointerProvenance(ModuleOp module)
{
    KernelPointerProvenance result;
    module.walk([&](FunctionOpInterface function) {
        if (function.isExternal() ||
            function.getOperation()->hasAttr("sym_visibility") &&
                function.getOperation()->getAttrOfType<StringAttr>("sym_visibility").getValue() == "private")
            return;
        SmallVector<int8_t, 8> pointers;
        for (Type type : function.getArgumentTypes()) {
            int8_t pointerKind = 0;
            if (auto ptr = dyn_cast<::tla::PtrType>(type))
                pointerKind = ptr.getAddrspace() == ::AddressSpace::gm ? 1 : -1;
            else if (auto tensor = dyn_cast<::tla::TlaTensorType>(type))
                pointerKind = tensor.getPtr().getAddrspace() == ::AddressSpace::gm ? 1 : -1;
            pointers.push_back(pointerKind);
        }
        result[function.getName()] = std::move(pointers);
    });
    return result;
}

std::string printType(Type type)
{
    std::string text;
    llvm::raw_string_ostream os(text);
    type.print(os);
    return text;
}

std::optional<unsigned> scalarStorageSize(Type type)
{
    if (type.isIndex())
        return 8;
    if (auto integer = dyn_cast<IntegerType>(type))
        switch (integer.getWidth()) {
            case 1:
            case 8:
                return 1;
            case 16:
                return 2;
            case 32:
                return 4;
            case 64:
                return 8;
            default:
                return std::nullopt;
        }
    if (type.isF16() || type.isBF16())
        return 2;
    if (type.isF32())
        return 4;
    return std::nullopt;
}

std::optional<py::dict> scalarAbiDescriptor(Type type)
{
    py::dict descriptor;
    if (type.isIndex()) {
        descriptor["category"] = "index";
        descriptor["bit_width"] = 64;
        descriptor["integer_signedness"] = py::none();
        descriptor["float_format"] = py::none();
        return descriptor;
    }
    if (auto integer = dyn_cast<IntegerType>(type)) {
        unsigned width = integer.getWidth();
        if (width != 1 && width != 8 && width != 16 && width != 32 && width != 64)
            return std::nullopt;
        descriptor["category"] = "integer";
        descriptor["bit_width"] = width;
        descriptor["integer_signedness"] = integer.isSigned()   ? "signed" :
                                           integer.isUnsigned() ? "unsigned" :
                                                                  "signless";
        descriptor["float_format"] = py::none();
        return descriptor;
    }
    StringRef format;
    unsigned width = 0;
    if (type.isF16()) {
        format = "f16";
        width = 16;
    } else if (type.isBF16()) {
        format = "bf16";
        width = 16;
    } else if (type.isF32()) {
        format = "f32";
        width = 32;
    } else {
        return std::nullopt;
    }
    descriptor["category"] = "float";
    descriptor["bit_width"] = width;
    descriptor["integer_signedness"] = py::none();
    descriptor["float_format"] = format.str();
    return descriptor;
}

static void appendAbiArgument(
    py::list& arguments, unsigned abiIndex, unsigned logicalIndex, const char* kind, std::optional<py::dict> scalar,
    const std::string& mlirType, uint64_t& offset, unsigned storageSize, const char* field = nullptr)
{
    offset = (offset + 3) & ~uint64_t(3);
    py::dict argument;
    argument["index"] = abiIndex;
    argument["logical_index"] = logicalIndex;
    argument["kind"] = kind;
    argument["scalar"] = scalar.has_value() ? py::object(*scalar) : py::none();
    argument["mlir_type"] = mlirType;
    argument["offset"] = offset;
    argument["storage_size"] = storageSize;
    argument["alignment"] = 4;
    if (field != nullptr)
        argument["field"] = field;
    else
        argument["field"] = py::none();
    arguments.append(argument);
    offset += storageSize;
}

static bool appendDynamicGmMemrefFields(
    py::list& arguments, unsigned& abiIndex, unsigned logicalIndex, MemRefType memrefType, uint64_t& offset)
{
    // Schema v4: unified 13-slot descriptor for all dynamic GM ranks.
    // Device signature is unified dynamic GM memref + originShape0/1 index args.
    SmallVector<int64_t, 4> strides;
    int64_t memrefOffset;
    if (memrefType.getRank() != 4 || !llvm::all_of(memrefType.getShape(), ShapedType::isDynamic) ||
        failed(getStridesAndOffset(memrefType, strides, memrefOffset)) || !ShapedType::isDynamic(memrefOffset) ||
        !llvm::all_of(strides, ShapedType::isDynamic))
        return false;
    std::string mlirType = printType(memrefType);
    static constexpr const char* kFields[] = {"allocated", "aligned",      "offset",      "size0",   "size1",
                                              "size2",     "size3",        "stride0",     "stride1", "stride2",
                                              "stride3",   "originShape0", "originShape1"};
    for (const char* field : kFields) {
        appendAbiArgument(
            arguments, abiIndex++, logicalIndex, "memref_field", std::nullopt, mlirType, offset, /*storageSize=*/8,
            field);
    }
    return true;
}

// The supported hivmc-a5 pipeline falls back to descriptors for the entire entry
// when a GM memref cannot use bare pointers. Shape, strides and offset must all
// be static for bare-pointer conversion.
static FailureOr<bool> requiresGmMemrefDescriptor(MemRefType memrefType)
{
    SmallVector<int64_t, 4> strides;
    int64_t memrefOffset;
    if (failed(getStridesAndOffset(memrefType, strides, memrefOffset)))
        return failure();
    return !memrefType.hasStaticShape() || ShapedType::isDynamic(memrefOffset) ||
           llvm::any_of(strides, ShapedType::isDynamic);
}

static bool appendGmMemrefFields(
    py::list& arguments, unsigned& abiIndex, unsigned logicalIndex, MemRefType memrefType, uint64_t& offset)
{
    unsigned rank = memrefType.getRank();
    if (rank == 0 || rank > 4)
        return false;

    std::string mlirType = printType(memrefType);
    appendAbiArgument(
        arguments, abiIndex++, logicalIndex, "memref_field", std::nullopt, mlirType, offset, /*storageSize=*/8,
        "allocated");
    appendAbiArgument(
        arguments, abiIndex++, logicalIndex, "memref_field", std::nullopt, mlirType, offset, /*storageSize=*/8,
        "aligned");
    appendAbiArgument(
        arguments, abiIndex++, logicalIndex, "memref_field", std::nullopt, mlirType, offset, /*storageSize=*/8,
        "offset");
    SmallVector<std::string, 8> fields;
    fields.reserve(2 * rank);
    for (unsigned dim = 0; dim < rank; ++dim)
        fields.push_back((Twine("size") + Twine(dim)).str());
    for (unsigned dim = 0; dim < rank; ++dim)
        fields.push_back((Twine("stride") + Twine(dim)).str());
    for (const std::string& field : fields)
        appendAbiArgument(
            arguments, abiIndex++, logicalIndex, "memref_field", std::nullopt, mlirType, offset, /*storageSize=*/8,
            field.c_str());
    return true;
}

std::optional<py::dict> buildKernelAbi(ModuleOp module, const KernelPointerProvenance& provenance)
{
    SmallVector<py::dict, 2> layouts;
    bool sawMixAic = false;
    bool sawMixAiv = false;
    std::optional<std::string> mixAicBase;
    std::optional<std::string> mixAivBase;
    for (func::FuncOp function : module.getOps<func::FuncOp>()) {
        if (function.isDeclaration() || !function->hasAttr("hacc.entry"))
            continue;
        StringRef name = function.getSymName();
        StringRef logicalName = name;
        if (name.ends_with("_mix_aic")) {
            sawMixAic = true;
            mixAicBase = name.drop_back(8).str();
        }
        if (name.ends_with("_mix_aiv")) {
            sawMixAiv = true;
            mixAivBase = name.drop_back(8).str();
        }
        if (name.ends_with("_mix_aic") || name.ends_with("_mix_aiv"))
            logicalName = name.drop_back(8);
        auto provenanceIt = provenance.find(logicalName);
        ArrayRef<int8_t> pointerArgs;
        if (provenanceIt != provenance.end())
            pointerArgs = provenanceIt->second;
        if (function.getNumResults() != 0)
            return std::nullopt;

        py::list arguments;
        uint64_t offset = 0;
        bool supported = true;
        unsigned logicalIndex = 0;
        unsigned abiIndex = 0;
        unsigned skipOriginIndexArgs = 0;
        ArrayRef<Type> argTypes = function.getArgumentTypes();
        bool requiresMemrefDescriptors = false;
        for (Type type : argTypes) {
            auto memrefType = dyn_cast<MemRefType>(type);
            if (!memrefType || !::tla::isGmMemRef(memrefType))
                continue;
            FailureOr<bool> requiresDescriptor = requiresGmMemrefDescriptor(memrefType);
            if (failed(requiresDescriptor))
                return std::nullopt;
            requiresMemrefDescriptors |= *requiresDescriptor;
        }
        bool anyMemrefField = false;
        for (auto [index, type] : llvm::enumerate(argTypes)) {
            if (function.getArgAttr(index, "tla.simt_print.workspace") ||
                function.getArgAttr(index, "tla.debug_print.workspace") ||
                function.getArgAttr(index, "tla.print_tensor.workspace") ||
                function.getArgAttr(index, "tla.debug_tunnel.state"))
                continue;
            // originShape0/1 are already folded into the 13-slot memref_field list.
            if (skipOriginIndexArgs > 0) {
                if (!type.isIndex()) {
                    supported = false;
                    break;
                }
                --skipOriginIndexArgs;
                continue;
            }
            unsigned provenanceIndex = logicalIndex;
            if (provenanceIndex < pointerArgs.size() && pointerArgs[provenanceIndex] < 0) {
                supported = false;
                break;
            }
            if (function.getArgAttr(index, "tla.dynamic_gm")) {
                auto memrefType = dyn_cast<MemRefType>(type);
                if (!memrefType || !::tla::isGmMemRef(memrefType) ||
                    !appendDynamicGmMemrefFields(arguments, abiIndex, logicalIndex, memrefType, offset)) {
                    supported = false;
                    break;
                }
                // Expect the next two function args to be originShape index companions.
                if (index + 2 >= argTypes.size() || !argTypes[index + 1].isIndex() || !argTypes[index + 2].isIndex()) {
                    supported = false;
                    break;
                }
                skipOriginIndexArgs = 2;
                anyMemrefField = true;
                ++logicalIndex;
                continue;
            }
            if (requiresMemrefDescriptors) {
                if (auto memrefType = dyn_cast<MemRefType>(type); memrefType && ::tla::isGmMemRef(memrefType)) {
                    if (!appendGmMemrefFields(arguments, abiIndex, logicalIndex, memrefType, offset)) {
                        supported = false;
                        break;
                    }
                    anyMemrefField = true;
                    ++logicalIndex;
                    continue;
                }
            }
            bool pointer = isa<MemRefType, LLVM::LLVMPointerType>(type) ||
                           (provenanceIndex < pointerArgs.size() && pointerArgs[provenanceIndex] > 0);
            unsigned size = 0;
            std::optional<py::dict> scalar;
            if (pointer) {
                size = 8;
            } else if (auto scalarSize = scalarStorageSize(type)) {
                size = *scalarSize;
                scalar = scalarAbiDescriptor(type);
            } else {
                supported = false;
                break;
            }
            if (!pointer && !scalar) {
                supported = false;
                break;
            }
            appendAbiArgument(
                arguments, abiIndex++, logicalIndex++, pointer ? "pointer" : "scalar", scalar, printType(type), offset,
                size);
        }
        if (!supported || skipOriginIndexArgs != 0)
            return std::nullopt;
        offset = (offset + 7) & ~uint64_t(7);
        py::dict layout;
        // schema v4 when any GM memref_field is present; otherwise keep v3
        // for backward-compatible static pointer/scalar layouts.
        layout["schema_version"] = anyMemrefField ? 4 : 3;
        layout["entrypoint"] = logicalName.str();
        layout["total_size"] = offset;
        layout["arguments"] = arguments;
        layouts.push_back(layout);
    }
    if (layouts.empty())
        return std::nullopt;
    if (layouts.size() > 1) {
        if (layouts.size() != 2 || !sawMixAic || !sawMixAiv || mixAicBase != mixAivBase)
            throw std::runtime_error(
                "Mixed kernel ABI collection expected exactly one AIC/AIV split "
                "entrypoint pair.");
        std::string expected = py::str(layouts.front()["arguments"]);
        for (const py::dict& layout : llvm::drop_begin(layouts))
            if (std::string(py::str(layout["arguments"])) != expected)
                throw std::runtime_error(
                    "Mixed AIC/AIV kernel ABI mismatch: split entrypoints have "
                    "different logical argument layouts.");
    }
    return layouts.front();
}

// Capture source facts before lowering erases PrintTensorOp. VF records use
// their own self-describing protocol and must not enter the legacy FIFO bound.
py::dict collectPrintTensorMetadata(ModuleOp module)
{
    py::dict records;
    module.walk([&](::tla::FuncOp function) {
        py::list tensors;
        function.walk([&](::tla::PrintTensorOp op) {
            if (op->getParentOfType<::tla::VecFuncOp>())
                return;
            auto ptr = op.getValue().getType().getPtr();
            py::dict tensor;
            tensor["shape"] = std::vector<int64_t>(op.getShape().begin(), op.getShape().end());
            APInt length;
            bool constant = matchPattern(op.getLength(), m_ConstantInt(&length));
            tensor["count"] = constant ? py::cast(length.getSExtValue()) : py::none();
            tensor["dtype"] = printType(ptr.getPointee());
            tensor["position"] = stringifyAddressSpace(ptr.getAddrspace()).str();
            tensor["call"] = py::len(tensors);
            tensor["dynamic_execution"] = !constant || op->hasAttr("dynamic_execution") ||
                                          llvm::any_of(op.getShape(), [](int64_t extent) { return extent < 0; });
            tensors.append(tensor);
        });
        records[py::str(function.getSymName().str())] = tensors;
    });
    return records;
}

py::dict buildRuntimeMetadata(ModuleOp module, StringRef entrypoint, const py::dict& sourceTensors)
{
    py::dict result;
    result["schema_version"] = 2;
    result["entrypoint"] = entrypoint.str();
    auto target = module->getAttrOfType<hacc::TargetAttr>(hacc::TargetAttr::name);
    if (!target || target.getTarget().getValue() != "Ascend950PR_9589")
        throw std::runtime_error("Unsupported or missing kernel target in runtime metadata.");
    result["target_arch"] = "c310";

    SmallVector<func::FuncOp> entries;
    for (auto function : module.getOps<func::FuncOp>())
        if (!function.isDeclaration() && function->hasAttr("hacc.entry"))
            entries.push_back(function);
    bool mixed = entries.size() == 2;
    int32_t aivSubblockCount = 1;
    bool scalar = false, tensor = false, simt = false, tunnel = false;
    SmallVector<StringRef> layout;
    std::string mode;
    for (auto [index, function] : llvm::enumerate(entries)) {
        auto core = function->getAttrOfType<hivm::TFuncCoreTypeAttr>(hivm::TFuncCoreTypeAttr::name);
        if (!core ||
            (core.getFuncCoreType() != hivm::TFuncCoreType::AIC && core.getFuncCoreType() != hivm::TFuncCoreType::AIV))
            throw std::runtime_error("Missing concrete entrypoint core type in runtime metadata.");
        mode = core.getFuncCoreType() == hivm::TFuncCoreType::AIC ? "aic" : "aiv";
        if (mixed) {
            auto ratio = function->getAttrOfType<hivm::TCoreRatioAttr>(hivm::TCoreRatioAttr::name);
            if (!ratio || ratio.getCube() != 1 || (ratio.getVector() != 1 && ratio.getVector() != 2))
                throw std::runtime_error("Missing or unsupported MIX core ratio in runtime metadata.");
            if (index != 0 && aivSubblockCount != ratio.getVector())
                throw std::runtime_error("Mixed entrypoints have different core ratios.");
            aivSubblockCount = ratio.getVector();
        }
        SmallVector<StringRef> functionLayout;
        if (auto attr = function->getAttrOfType<StringAttr>("tla.print.layout"))
            attr.getValue().split(functionLayout, ',');
        if (index == 0)
            layout = functionLayout;
        else if (layout != functionLayout)
            throw std::runtime_error("Mixed entrypoints have different print context layouts.");
        SmallVector<StringRef> actualLayout;
        for (unsigned argument = 0; argument < function.getNumArguments(); ++argument) {
            StringRef role = "user";
            if (function.getArgAttr(argument, "tla.debug_print.workspace") ||
                function.getArgAttr(argument, "tla.print_tensor.workspace"))
                role = "fifo";
            else if (function.getArgAttr(argument, "tla.simt_print.workspace"))
                role = "simt";
            else if (function.getArgAttr(argument, "tla.debug_tunnel.state"))
                role = "tunnel";
            if (actualLayout.empty() || role != "user" || actualLayout.back() != "user")
                actualLayout.push_back(role);
            scalar |= static_cast<bool>(function.getArgAttr(argument, "tla.debug_print.workspace"));
            tensor |= static_cast<bool>(function.getArgAttr(argument, "tla.print_tensor.workspace"));
            simt |= static_cast<bool>(function.getArgAttr(argument, "tla.simt_print.workspace"));
            tunnel |= static_cast<bool>(function.getArgAttr(argument, "tla.debug_tunnel.state"));
        }
        // Zero user arguments still have a logical insertion point in layout.
        if (!functionLayout.empty()) {
            if (!llvm::is_contained(actualLayout, "user")) {
                auto user = llvm::find(functionLayout, "user");
                if (user != functionLayout.end()) {
                    auto index = user - functionLayout.begin();
                    if (static_cast<size_t>(index) > actualLayout.size())
                        throw std::runtime_error("Invalid user insertion point in print context layout.");
                    actualLayout.insert(actualLayout.begin() + index, "user");
                }
            }
            if (actualLayout != functionLayout)
                throw std::runtime_error("Print context layout does not match the entrypoint arguments.");
        }
    }
    result["kernel_mode"] = mixed ? "mix" : mode;
    result["aiv_subblock_count"] = aivSubblockCount;
    result["uses_scalar_print"] = scalar;
    result["uses_tensor_print"] = tensor;
    result["uses_debug_tunnel"] = tunnel;
    py::list contextLayout;
    for (StringRef role : layout)
        contextLayout.append(role.str());
    result["print_argument_layout"] = contextLayout;

    bool aicTensor = false, aivTensor = false, l1 = false, l0c = false, simtScalar = false;
    module.walk([&](func::CallOp call) {
        StringRef callee = call.getCallee();
        callee.consume_front("_mlir_ciface_");
        simtScalar |= callee == "tla_simt_printf";
        if (!callee.consume_front("tla_print_tensor_"))
            return;
        auto function = call->getParentOfType<func::FuncOp>();
        auto core = function->getAttrOfType<hivm::TFuncCoreTypeAttr>(hivm::TFuncCoreTypeAttr::name);
        if (!core)
            throw std::runtime_error("Tensor print helper has no enclosing core type.");
        aicTensor |= core.getFuncCoreType() == hivm::TFuncCoreType::AIC;
        aivTensor |= core.getFuncCoreType() == hivm::TFuncCoreType::AIV;
        l1 |= callee.starts_with("l1_");
        l0c |= callee.starts_with("l0c_");
    });
    result["print_helper_core"] = aicTensor && aivTensor ? py::cast("both") :
                                  aicTensor              ? py::cast("aic") :
                                  aivTensor              ? py::cast("aiv") :
                                                           py::none();
    py::list positions;
    if (l1)
        positions.append("L1");
    if (l0c)
        positions.append("L0C");
    result["local_print_positions"] = positions;
    auto formats = module->getAttrOfType<StringAttr>("tla.simt_print.formats");
    if (simtScalar && !formats)
        throw std::runtime_error("SIMT scalar print format metadata is missing.");
    result["simt_print_formats"] = simt ? py::cast(formats ? formats.getValue().str() : "") : py::none();
    bool sharedRecords = scalar || simt || (aicTensor && aivTensor);
    auto key = py::str(entrypoint.str());
    result["tensor_records"] = py::list();
    if (tensor && !sharedRecords && sourceTensors.contains(key))
        result["tensor_records"] = sourceTensors[key];
    return result;
}

py::dict lowerToMlir(
    MlirModule cModule, std::vector<std::string> printBefore, std::vector<std::string> printAfter, bool printBeforeAll,
    bool printAfterAll)
{
    ModuleOp module = moduleFromCapsule(cModule);
    ::tla::registerTlaPasses();
    MLIRContext* context = module.getContext();
    context->allowUnregisteredDialects(true);
    context->disableMultithreading();
    ::tla::tools::loadTlaCompileDialects(*context);

    PassManager tlaPm(context);
    ::tla::tools::buildTlaCompilePassManagers(*context, tlaPm);
    KernelPointerProvenance pointerProvenance = collectKernelPointerProvenance(module);
    py::dict sourceTensors = collectPrintTensorMetadata(module);

    std::string passDump;
    llvm::raw_string_ostream passDumpStream(passDump);
    if (printBeforeAll || printAfterAll || !printBefore.empty() || !printAfter.empty()) {
        auto shouldPrintBefore = [printBeforeAll, printBefore](Pass* pass, Operation*) {
            return printBeforeAll || containsPassName(printBefore, pass);
        };
        auto shouldPrintAfter = [printAfterAll, printAfter](Pass* pass, Operation*) {
            return printAfterAll || containsPassName(printAfter, pass);
        };
        tlaPm.enableIRPrinting(
            shouldPrintBefore, shouldPrintAfter,
            /*printModuleScope=*/true,
            /*printAfterOnlyOnChange=*/false,
            /*printAfterOnlyOnFailure=*/false, passDumpStream);
    }

    std::string output;
    std::string error;
    std::vector<BridgeDiagnostic> diagnostics;
    ScopedDiagnosticHandler diagnosticHandler(context, [&](Diagnostic& diagnostic) {
        captureDiagnosticRecords(diagnostic, diagnostics);
        // Errors are transported through BridgeLoweringError. Non-errors are
        // deliberately forwarded to pre-existing diagnostic handlers, which
        // retain ownership of their presentation on successful lowerings.
        return diagnostic.getSeverity() == DiagnosticSeverity::Error ? success() : failure();
    });
    bool success =
        ::tla::tools::runTlaCompilePipelinesWithManagers(module, StringRef("mlir"), tlaPm, output, error, true);
    passDumpStream.flush();
    py::dict result;
    result["success"] = success;
    result["error"] = success ? "" : (error.empty() ? "Failed to run Tla pipeline." : error);
    py::list diagnosticRecords;
    for (const BridgeDiagnostic& diagnostic : diagnostics)
        diagnosticRecords.append(diagnosticToPython(diagnostic));
    result["diagnostics"] = std::move(diagnosticRecords);
    result["lowered_mlir"] = output;
    result["pass_ir_dump"] = passDump;
    auto moduleU64 = [&](StringRef name) -> uint64_t {
        auto attr = module->getAttrOfType<IntegerAttr>(name);
        return attr ? attr.getValue().getZExtValue() : 0;
    };
    auto moduleI64 = [&](StringRef name, int64_t fallback) -> int64_t {
        auto attr = module->getAttrOfType<IntegerAttr>(name);
        return attr ? attr.getValue().getSExtValue() : fallback;
    };
    // What the compiler placed. Named for what it is: the launch-sized region is
    // not part of it, so a bare "ub_bytes" reads as the kernel's whole footprint.
    result["ub_static_bytes"] = moduleU64("tla.ub_static_bytes");
    // The ceiling this kernel is held to. It already accounts for the mode and
    // for a released compiler reserve, so the host never recomputes it.
    result["ub_programmable_bytes"] = moduleU64("tla.ub_programmable_bytes");
    // Where the launch-sized region starts, or -1 for a kernel that declares
    // none. Zero is a real base -- a kernel can have a region and no static
    // allocations at all -- so absence needs a value outside the range.
    result["ub_dynamic_base"] = moduleI64("tla.ub_dynamic_base", -1);

    // A failed pass may leave partially rewritten functions behind. ABI
    // collection is meaningful only for successful lowering and must never
    // replace the compiler diagnostic that caused the failure.
    if (success) {
        if (auto kernelAbi = buildKernelAbi(module, pointerProvenance)) {
            result["kernel_abi"] = *kernelAbi;
            result["runtime_metadata"] =
                buildRuntimeMetadata(module, py::cast<std::string>((*kernelAbi)["entrypoint"]), sourceTensors);
        } else {
            result["kernel_abi"] = py::none();
            result["runtime_metadata"] = py::none();
        }
    } else {
        result["kernel_abi"] = py::none();
    }
    return result;
}

} // namespace

PYBIND11_MODULE(_tla_type_bridge_native, m)
{
    m.doc() = "Native Tla TypeDef construction/accessors for Python MLIR types.";

    m.def("load_tla_dialect", &loadTlaDialect, py::arg("context"));

    m.def("shape_type_get", &shapeTypeGet, py::arg("context"), py::arg("tree"));
    m.def("coord_type_get", &coordTypeGet, py::arg("context"), py::arg("tree"));
    m.def("stride_type_get", &strideTypeGet, py::arg("context"), py::arg("tree"));

    m.def(
        "layout_type_get", &layoutTypeGet, py::arg("context"), py::arg("shape_tree"), py::arg("stride_tree"),
        py::arg("origin_tree") = py::none(), py::arg("layout") = "RowMajor");
    m.def(
        "layout_type_from_components_get", &layoutTypeFromComponentsGet, py::arg("context"), py::arg("shape_type"),
        py::arg("stride_type"), py::arg("origin_shape_type") = py::none(), py::arg("layout") = "RowMajor");
    m.def(
        "tensor_type_get", &tensorTypeGet, py::arg("context"), py::arg("shape_tree"), py::arg("stride_tree"),
        py::arg("coord_tree"), py::arg("origin_shape_tree"), py::arg("element_type"), py::arg("addrspace"),
        py::arg("layout"), py::arg("ptr_alignment"));
    m.def(
        "dynamic_gm_memref_type", &dynamicGmMemrefTypeGet, py::arg("tensor_type"),
        "Build the unified dynamic GM memref type for schema-v4 ABI from a !tla.tensor.");
    m.def(
        "ptr_type_get", &ptrTypeGet, py::arg("context"), py::arg("pointee"), py::arg("addrspace"),
        py::arg("alignment"));
    m.def(
        "vector_ssa_type_get", &vectorSSATypeGet, py::arg("context"), py::arg("valid_lanes"), py::arg("element_type"));
    m.def("mask_ssa_type_get", &maskSSATypeGet, py::arg("context"), py::arg("physical_lanes"));
    m.def("flag_type_get", &flagTypeGet, py::arg("context"));
    m.def("cross_flag_type_get", &crossFlagTypeGet, py::arg("context"), py::arg("mode"));
    m.def("cross_flag_mode", &crossFlagMode, py::arg("type"));
    m.def("mutex_type_get", &mutexTypeGet, py::arg("context"));
    m.def("float4_e2m1_type_get", &float4E2M1TypeGet, py::arg("context"));
    m.def("float4_e1m2_type_get", &float4E1M2TypeGet, py::arg("context"));
    m.def("float8_e8m0_type_get", &float8E8M0TypeGet, py::arg("context"));
    m.def("copy_l0c2dst_params_type_get", &copyL0C2DstParamsTypeGet, py::arg("context"));

    m.def("type_is_ptr", &typeIs<::tla::PtrType>, py::arg("type"));
    m.def("type_is_tensor", &typeIs<::tla::TlaTensorType>, py::arg("type"));
    m.def("type_is_shape", &typeIs<::tla::ShapeType>, py::arg("type"));
    m.def("type_is_coord", &typeIs<::tla::CoordType>, py::arg("type"));
    m.def("type_is_stride", &typeIs<::tla::StrideType>, py::arg("type"));
    m.def("type_is_layout", &typeIs<::tla::LayoutType>, py::arg("type"));
    m.def("type_is_vector_ssa", &typeIs<::tla::VectorSSAType>, py::arg("type"));
    m.def("type_is_mask_ssa", &typeIs<::tla::MaskSSAType>, py::arg("type"));
    m.def("type_is_flag", &typeIs<::tla::FlagType>, py::arg("type"));
    m.def("type_is_cross_flag", &typeIs<::tla::CrossFlagType>, py::arg("type"));
    m.def("type_is_mutex", &typeIs<::tla::MutexType>, py::arg("type"));
    m.def("type_is_copy_l0c2dst_params", &typeIs<::tla::CopyL0C2DstParamsType>, py::arg("type"));
    m.def("tla_type_category", &tlaTypeCategory, py::arg("type"));

    m.def("ptr_pointee_type_get", &ptrPointeeTypeGet, py::arg("ptr_type"));
    m.def("ptr_addrspace", &ptrAddrspace, py::arg("ptr_type"));
    m.def("ptr_alignment", &ptrAlignment, py::arg("ptr_type"));
    m.def("tensor_ptr_type_get", &tensorPtrTypeGet, py::arg("tensor_type"));
    m.def("vector_ssa_element_type_get", &vectorSSAElementTypeGet, py::arg("vector_type"));
    m.def("vector_ssa_valid_lanes_get", &vectorSSAValidLanesGet, py::arg("vector_type"));
    m.def("mask_ssa_physical_lanes_get", &maskSSAPhysicalLanesGet, py::arg("mask_type"));
    m.def(
        "lower_to_mlir", &lowerToMlir, py::arg("module"), py::arg("mlir_print_ir_before"),
        py::arg("mlir_print_ir_after"), py::arg("mlir_print_ir_before_all"), py::arg("mlir_print_ir_after_all"),
        "Lower an mlir.ir.Module through the typed MLIR Python bridge.");
}
