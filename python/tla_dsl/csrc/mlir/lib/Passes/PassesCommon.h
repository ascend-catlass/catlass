#pragma once

#include "Dialect/Tla/IR/TlaAttrs.h"
#include "Dialect/Tla/IR/TlaOps.h"
#include "Dialect/Tla/IR/TlaTypes.h"
#include "Tools/AddressSpaceConversion.h"

#include "Passes.h"
#include "mlir/AsmParser/AsmParser.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/DLTI/DLTI.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Types.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/raw_ostream.h"
#include <cstdlib>
#include <functional>
#include <limits>
#include <optional>
#include <utility>

#include "bishengir/Dialect/HACC/IR/HACC.h"
#include "bishengir/Dialect/HACC/Utils/Utils.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/HIVMRegbaseIntrins/IR/HIVMRegbaseIntrins.h"
using namespace mlir;

namespace tla {

inline constexpr StringLiteral kDltiTargetSystemSpecAttrName = "dlti.target_system_spec";
inline constexpr StringLiteral kMixModeAttrName = "mix_mode";
inline constexpr StringLiteral kParallelModeAttrName = "parallel_mode";
inline constexpr StringLiteral kHivmVectorFunctionAttrName = "hivm.vector_function";
inline constexpr StringLiteral kCoreRatioAttrName = "tla.core_ratio";

inline hivm::TModuleCoreType toModuleCoreType(hivm::TFuncCoreType coreType)
{
    switch (coreType) {
        case hivm::TFuncCoreType::AIC:
            return hivm::TModuleCoreType::AIC;
        case hivm::TFuncCoreType::AIV:
            return hivm::TModuleCoreType::AIV;
        case hivm::TFuncCoreType::MIX:
            return hivm::TModuleCoreType::MIX;
        case hivm::TFuncCoreType::AIC_OR_AIV:
            break;
    }
    llvm_unreachable("AIC_OR_AIV has no module core type equivalent");
}

// The function's effective core type, resolved from the hivm.func_core_type
// attribute stamped by TlaLowerFuncPass (and carried onto outlined fragments
// by the tla.vector lowering passes / tla-cube-region / tla-split-mixed-func).
// This is the
// single carrier of the per-function AIC/AIV/MIX classification through the
// lowering pipeline;
inline std::optional<hivm::TFuncCoreType> getFunctionCoreType(Operation* op)
{
    if (!op)
        return std::nullopt;
    auto attr = op->getAttrOfType<hivm::TFuncCoreTypeAttr>(hivm::TFuncCoreTypeAttr::name);
    if (!attr)
        return std::nullopt;
    return attr.getFuncCoreType();
}

inline bool isPrivateSymbol(Operation* op)
{
    if (auto visibility = op->getAttrOfType<StringAttr>(SymbolTable::getVisibilityAttrName()))
        return visibility.getValue() == "private";
    return false;
}

inline bool hasC310TargetAttrs(ModuleOp module)
{
    auto targetAttr = module->getAttrOfType<hacc::TargetAttr>(hacc::TargetAttr::name);
    return targetAttr && targetAttr.getTarget().getValue() == "Ascend950PR_9589" &&
           module->hasAttr(kDltiTargetSystemSpecAttrName);
}

inline void ensureC310TargetAttrs(ModuleOp module)
{
    MLIRContext* ctx = module.getContext();
    OpBuilder builder(ctx);
    module->setAttr(hacc::TargetAttr::name, hacc::TargetAttr::get(ctx, StringAttr::get(ctx, "Ascend950PR_9589")));

    SmallVector<std::pair<StringRef, Attribute>, 14> specs = {
        {"AI_CORE_COUNT", builder.getI32IntegerAttr(32)},
        {"CUBE_CORE_COUNT", builder.getI32IntegerAttr(32)},
        {"VECTOR_CORE_COUNT", builder.getI32IntegerAttr(64)},
        // *_SIZE values are computed as bits
        {"UB_SIZE", builder.getI32IntegerAttr(248 * 1024 * 8)},
        {"L1_SIZE", builder.getI32IntegerAttr(512 * 1024 * 8)},
        {"L0A_SIZE", builder.getI32IntegerAttr(64 * 1024 * 8)},
        {"L0B_SIZE", builder.getI32IntegerAttr(64 * 1024 * 8)},
        {"L0C_SIZE", builder.getI32IntegerAttr(256 * 1024 * 8)},
        {"UB_ALIGN_SIZE", builder.getI32IntegerAttr(32 * 8)},
        {"L1_ALIGN_SIZE", builder.getI32IntegerAttr(32 * 8)},
        {"L0C_ALIGN_SIZE", builder.getI32IntegerAttr(1024 * 8)},
        {"MINIMAL_D_CACHE_SIZE", builder.getI32IntegerAttr(32 * 1024 * 8)},
        {"MAXIMUM_D_CACHE_SIZE", builder.getI32IntegerAttr(120 * 1024 * 8)},
        {"ARCH", builder.getStringAttr("dav-c310")},
    };
    SmallVector<DataLayoutEntryInterface> entries;
    for (auto [name, value] : specs) {
        entries.push_back(DataLayoutEntryAttr::get(builder.getStringAttr(name), value));
    }

    auto targetSpec = cast<hacc::HACCTargetDeviceSpecInterface>(hacc::TargetDeviceSpecAttr::get(ctx, entries));
    hacc::utils::setNPUTargetSpec(module, targetSpec);
}

inline bool hasRequiredHaccEntryAttrs(Operation* op)
{
    auto functionKind = op->getAttrOfType<hacc::HACCFuncTypeAttr>(hacc::HACCFuncTypeAttr::name);
    return op->hasAttr(hacc::stringifyEnum(hacc::HACCToLLVMIRTranslateAttr::ENTRY)) && functionKind &&
           functionKind.getFunctionKind() == hacc::HACCFuncType::DEVICE;
}

inline void setRequiredHaccEntryAttrs(Operation* op, MLIRContext* ctx)
{
    op->setAttr(hacc::stringifyEnum(hacc::HACCToLLVMIRTranslateAttr::ENTRY), UnitAttr::get(ctx));
    op->setAttr(hacc::HACCFuncTypeAttr::name, hacc::HACCFuncTypeAttr::get(ctx, hacc::HACCFuncType::DEVICE));
}

inline void setC310RegbaseTargetAttr(Operation* op, MLIRContext* ctx)
{
    op->setAttr(
        StringAttr::get(ctx, hivm_regbaseintrins::kDavinciTargetAttrName),
        hivm_regbaseintrins::SIMT_TargetAttr::get(ctx, "dav-c310"));
}

struct LowerTlaReturnToFuncReturnPattern : public OpRewritePattern<::tla::ReturnOp> {
    using OpRewritePattern<::tla::ReturnOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(::tla::ReturnOp op, PatternRewriter& rewriter) const override
    {
        rewriter.replaceOpWithNewOp<func::ReturnOp>(op);
        return success();
    }
};

inline FailureOr<hivm::AddressSpace> mapTlaAddressSpaceToHivm(AddressSpace addressSpace)
{
    switch (addressSpace) {
        case AddressSpace::generic:
            return hivm::AddressSpace::Zero;
        case AddressSpace::gm:
            return hivm::AddressSpace::GM;
        case AddressSpace::l1:
            return hivm::AddressSpace::L1;
        case AddressSpace::l0a:
            return hivm::AddressSpace::L0A;
        case AddressSpace::l0b:
            return hivm::AddressSpace::L0B;
        case AddressSpace::l0c:
            return hivm::AddressSpace::L0C;
        case AddressSpace::ub:
            return hivm::AddressSpace::UB;
        case AddressSpace::biasbuf:
            return hivm::AddressSpace::BiasBUF;
    }
    return failure();
}

inline FailureOr<Attribute> mapTlaAddressSpaceToHivmMemspace(MLIRContext* ctx, AddressSpace addressSpace)
{
    FailureOr<hivm::AddressSpace> hivmSpace = mapTlaAddressSpaceToHivm(addressSpace);
    if (failed(hivmSpace))
        return failure();
    return hivm::AddressSpaceAttr::get(ctx, *hivmSpace);
}

inline bool hasZeroStaticCoords(ArrayRef<int64_t> coords)
{
    return llvm::all_of(coords, [](int64_t coord) { return coord == 0; });
}

inline bool hasDefaultRowMajorStrides(ArrayRef<int64_t> shape, ArrayRef<int64_t> strides)
{
    if (shape.empty() || shape.size() != strides.size())
        return false;

    int64_t expectedStride = 1;
    for (size_t index = shape.size(); index-- > 0;) {
        if (shape[index] == ShapedType::kDynamic || strides[index] == ShapedType::kDynamic)
            return false;
        if (strides[index] != expectedStride)
            return false;
        if (shape[index] != 0 && expectedStride > std::numeric_limits<int64_t>::max() / shape[index])
            return false;
        expectedStride *= shape[index];
    }
    return true;
}

// The element type a memref is built over. The dialect's own cube element
// formats (f4e2m1 / f4e1m2 / f8e8m0) have no LLVM lowering, so a memref must
// never carry one; they are buffered as bytes. Every memref builder goes through
// here rather than reading the pointee directly -- getting this wrong shows up
// as "invalid memref element type" far from the cause.
//
// Only the element type changes, never the extents. That is safe because the GM
// tensors this applies to are layout-dynamic, so their memrefs are ?x?x?x? and
// carry no element count; the true counts travel beside them as origin_shape. A
// statically shaped sub-byte tensor would additionally need its contiguous
// extent halved, and does not arise today.
inline Type storageElementType(MLIRContext* ctx, Type elementType)
{
    return ::tla::isTlaCustomElementType(elementType) ? IntegerType::get(ctx, 8) : elementType;
}

inline FailureOr<MemRefType> buildHivmMemrefType(
    MLIRContext* ctx, ArrayRef<int64_t> shape, Type elementType, ::AddressSpace addrSpace)
{
    FailureOr<Attribute> memorySpaceOr = mapTlaAddressSpaceToHivmMemspace(ctx, addrSpace);
    if (failed(memorySpaceOr))
        return failure();
    return MemRefType::get(shape, storageElementType(ctx, elementType), AffineMap(), *memorySpaceOr);
}

inline FailureOr<MemRefType> bridgeTlaTensorStorageType(Type tlaTensorType)
{
    SmallVector<int64_t, 4> shape;
    SmallVector<int64_t, 4> strides;
    SmallVector<int64_t, 4> coords;
    SmallVector<int64_t, 4> originShape;
    std::string elementTypeStorage;
    std::string addressSpaceStorage;
    StringRef elementTypeText;
    StringRef addressSpace;
    ::LayoutTag layoutTag = ::LayoutTag::Unknown;

    auto tensorTy = dyn_cast<::tla::TlaTensorType>(tlaTensorType);
    if (!tensorTy)
        return failure();
    auto layout = tensorTy.getLayout();
    auto ptr = tensorTy.getPtr();
    if (!layout.getOrigin() || !ptr.getPointee())
        return failure();
    if (failed(::tla::getTlaIndexTreeLeaves(layout.getShape().getTree(), shape)) ||
        failed(::tla::getTlaIndexTreeLeaves(layout.getStride().getTree(), strides)) ||
        failed(::tla::getTlaIndexTreeLeaves(tensorTy.getCoord().getTree(), coords)) ||
        failed(::tla::getTlaIndexTreeLeaves(layout.getOrigin().getTree(), originShape)))
        return failure();
    llvm::raw_string_ostream os(elementTypeStorage);
    os << ptr.getPointee();
    os.flush();
    elementTypeText = StringRef(elementTypeStorage).trim();
    addressSpaceStorage = stringifyAddressSpace(ptr.getAddrspace()).str();
    addressSpace = addressSpaceStorage;
    layoutTag = layout.getLayoutTag();

    if (shape.empty() || strides.empty() || coords.empty() || originShape.empty() || elementTypeText.empty() ||
        addressSpace.empty() || layoutTag == ::LayoutTag::Unknown)
        return failure();

    if (layoutTag != ::LayoutTag::RowMajor && originShape.size() == coords.size() &&
        shape.size() != originShape.size()) {
        shape = originShape;
    }
    if (strides.size() != shape.size())
        strides = shape;
    if (coords.size() != shape.size()) {
        if (originShape.size() == coords.size() && shape.size() == originShape.size()) {
            // Keep the parsed coordinates.
        } else {
            return failure();
        }
    }

    MLIRContext* ctx = tlaTensorType.getContext();
    Type elementType = storageElementType(ctx, ptr.getPointee());
    auto tlaAddressSpace = symbolizeAddressSpace(addressSpace);
    if (!tlaAddressSpace)
        return failure();
    FailureOr<Attribute> memorySpaceOr = mapTlaAddressSpaceToHivmMemspace(ctx, *tlaAddressSpace);
    if (failed(memorySpaceOr))
        return failure();

    if (layoutTag == ::LayoutTag::RowMajor &&
        !(hasZeroStaticCoords(coords) && hasDefaultRowMajorStrides(shape, strides))) {
        auto layout = StridedLayoutAttr::get(ctx, ShapedType::kDynamic, strides);
        return MemRefType::get(shape, elementType, layout, *memorySpaceOr);
    }
    return MemRefType::get(shape, elementType, AffineMap(), *memorySpaceOr);
}

/// Whether `tla.func` → `func.func` lowering copies attributes other than `sym_name` /
/// `function_type` onto the new `func.func` (HACC pipeline needs them; std lowering omits).
enum class LowerTlaFuncToFuncAttrPolicy
{
    CopyNonSignatureAttrs,
    OmitAttrs
};

template <LowerTlaFuncToFuncAttrPolicy AttrPolicy>
struct LowerTlaFuncToFuncPattern : public OpRewritePattern<::tla::FuncOp> {
    using OpRewritePattern<::tla::FuncOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(::tla::FuncOp op, PatternRewriter& rewriter) const override
    {
        auto symNameAttr = op->getAttrOfType<StringAttr>("sym_name");
        auto typeAttr = op->getAttrOfType<TypeAttr>("function_type");
        if (!symNameAttr || !typeAttr) {
            op.emitError() << "expected tla.func to have sym_name and function_type";
            return failure();
        }

        auto funcType = llvm::dyn_cast<FunctionType>(typeAttr.getValue());
        if (!funcType) {
            op.emitError() << "expected tla.func function_type to be a FunctionType";
            return failure();
        }

        // Container lowering preserves the TLA signature. TlaLowerFuncPass owns
        // device-entry ABI conversion and root descriptor materialization after
        // validating every function in the module.
        auto func = rewriter.create<func::FuncOp>(op.getLoc(), symNameAttr.getValue(), funcType);
        if constexpr (AttrPolicy == LowerTlaFuncToFuncAttrPolicy::CopyNonSignatureAttrs) {
            for (NamedAttribute attr : op->getAttrs()) {
                StringRef name = attr.getName().getValue();
                if (name == "sym_name" || name == "function_type")
                    continue;
                func->setAttr(attr.getName(), attr.getValue());
            }
        }
        func.getBody().takeBody(op.getRegion());
        rewriter.eraseOp(op);
        return success();
    }
};

/// Cast integer/index values to i32 (HIVM `.post` stride / byte-offset operands).
inline Value castToI32(OpBuilder& b, Location loc, Value v)
{
    if (v.getType().isInteger(32))
        return v;
    if (isa<IndexType>(v.getType()))
        return b.create<arith::IndexCastOp>(loc, b.getI32Type(), v);
    if (auto it = dyn_cast<IntegerType>(v.getType()); it && it.getWidth() < 32)
        return b.create<arith::ExtSIOp>(loc, b.getI32Type(), v);
    if (auto it = dyn_cast<IntegerType>(v.getType()); it && it.getWidth() > 32)
        return b.create<arith::TruncIOp>(loc, b.getI32Type(), v);
    return v;
}

} // namespace tla
