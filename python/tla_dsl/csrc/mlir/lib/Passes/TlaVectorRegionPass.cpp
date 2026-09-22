#include "PassesCommon.h"
#include "PassesInternal.h"
#include "Passes/TlaTensorToMemref.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/HIVMAVE/IR/HIVMAVE.h"
#include "bishengir/Dialect/Utils/Util.h"
#include "llvm/ADT/StringSwitch.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include <algorithm>
#include <limits>

namespace tla {
namespace {
// ParsedTensorInfo + parseTensorInfo live in the shared header
// Passes/TlaTensorToMemref.h (raw, non-normalized decode). Unqualified uses below
// resolve to ::tla:: via namespace lookup.

static hivmave::VFPgeOp createAvePgeMask(OpBuilder& b, Location loc, VectorType maskType, hivmave::PgePattern pattern)
{
    return b.create<hivmave::VFPgeOp>(loc, maskType, pattern);
}

static hivmave::VFPltOp createAvePltMask(OpBuilder& b, Location loc, VectorType maskType, Value trueShape)
{
    return b.create<hivmave::VFPltOp>(loc, maskType, b.getIndexType(), trueShape);
}

static LogicalResult lowerLocalMemBar(OpBuilder& b, ::tla::LocalMemBarOp op)
{
    int64_t barrierKind = op.getBarrierKind();
    if (barrierKind < 0 || barrierKind > 11)
        return op.emitError("barrier_kind ") << barrierKind << " is out of range [0, 11]";
    Value encoded = b.create<arith::ConstantIntOp>(op.getLoc(), barrierKind, 32);
    b.create<hivmave::VFMemBarOp>(op.getLoc(), encoded);
    return success();
}

static hivmave::LoadDist mapTlaLoadDistToAve(::LoadDist dist)
{
    switch (dist) {
        case ::LoadDist::norm:
            return hivmave::LoadDist::NORM;
        case ::LoadDist::brc_b32:
            return hivmave::LoadDist::BRC_B32;
        case ::LoadDist::dintlv_b32:
            return hivmave::LoadDist::DINTLV_B32;
        case ::LoadDist::us_b8:
            return hivmave::LoadDist::US_B8;
        case ::LoadDist::brc_b16:
            return hivmave::LoadDist::BRC_B16;
        case ::LoadDist::us_b16:
            return hivmave::LoadDist::US_B16;
        case ::LoadDist::unpack_b16:
            return hivmave::LoadDist::UNPK_B16;
        case ::LoadDist::e2b_b16:
            return hivmave::LoadDist::E2B_B16;
        case ::LoadDist::e2b_b32:
            return hivmave::LoadDist::E2B_B32;
        case ::LoadDist::blk:
            return hivmave::LoadDist::BLK;
    }
    llvm_unreachable("unsupported tla.load load_dist");
}

static hivmave::StoreDist mapTlaStoreDistToAve(::StoreDist dist)
{
    switch (dist) {
        case ::StoreDist::pack_b32:
            return hivmave::StoreDist::PK_B32;
        case ::StoreDist::pack_b16:
            return hivmave::StoreDist::PK_B16;
        case ::StoreDist::first_element_b8:
            return hivmave::StoreDist::ONEPT_B8;
        case ::StoreDist::first_element_b16:
            return hivmave::StoreDist::ONEPT_B16;
        case ::StoreDist::first_element_b32:
            return hivmave::StoreDist::ONEPT_B32;
        // If other StoreDist added, complete here
        default:
            return hivmave::StoreDist::NORM_B8;
    }
}

static bool isDualDestLoadDist(hivmave::LoadDist pattern)
{
    return pattern == hivmave::LoadDist::DINTLV_B8 || pattern == hivmave::LoadDist::DINTLV_B16 ||
           pattern == hivmave::LoadDist::DINTLV_B32;
}

static bool isDualDestStoreDist(hivmave::StoreDist pattern)
{
    return pattern == hivmave::StoreDist::INTLV_B8 || pattern == hivmave::StoreDist::INTLV_B16 ||
           pattern == hivmave::StoreDist::INTLV_B32;
}

static hivmave::VFLoadOp createVFLoad(
    OpBuilder& b, Location loc, VectorType vecType, Value memref, Value index, hivmave::LoadDist pattern,
    bool unaligned)
{
    // Always pass ``pattern`` at create time. Dual-destination dists need two
    // result types up front (AVE's convenience builder only emits a single NORM
    // result), and single-destination non-NORM (e.g. BRC_B32) can use the same
    // pattern-in-create overload instead of create-then-setPattern.
    SmallVector<Type, 2> resultTypes;
    resultTypes.push_back(vecType);
    if (isDualDestLoadDist(pattern))
        resultTypes.push_back(vecType);

    auto load = b.create<hivmave::VFLoadOp>(loc, resultTypes, pattern, memref, ValueRange{index});
    if (unaligned)
        load->setAttr(hivmave::UnalignedAttr::name, hivmave::UnalignedAttr::get(b.getContext()));
    return load;
}

// Map a 1/2/4-byte MaskSSA UB tile memref onto an i1 view of ``lanes``
// predicate bits at the same byte address (AscendC MaskDist::DIST_NORM /
// plds/psts.b8). UB element width types the pointer; the hardware transfer
// is always the bit-level plds/psts.b8 path.
static FailureOr<Value> materializeI1MaskMemrefFromUb(OpBuilder& b, Location loc, Value ubMemref, int64_t lanes)
{
    if (lanes <= 0 || lanes % 8 != 0)
        return failure();
    auto srcTy = dyn_cast<MemRefType>(ubMemref.getType());
    if (!srcTy)
        return failure();
    int64_t elemBytes = getByteSizeOfFixedWidthScalarType(srcTy.getElementType());
    if (elemBytes != 1 && elemBytes != 2 && elemBytes != 4)
        return failure();

    auto meta = b.create<mlir::memref::ExtractStridedMetadataOp>(loc, ubMemref);
    Value basePtr = b.create<mlir::memref::ExtractAlignedPointerAsIndexOp>(loc, meta.getBaseBuffer());
    // ExtractStridedMetadata offset is in elements; convert to byte offset.
    Value elemOffset = meta.getOffset();
    Value byteOffset = elemOffset;
    if (elemBytes != 1) {
        Value scale = b.create<arith::ConstantIndexOp>(loc, elemBytes);
        byteOffset = b.create<arith::MulIOp>(loc, elemOffset, scale);
    }
    Value byteAddr = b.create<arith::AddIOp>(loc, basePtr, byteOffset);
    Value addrI64 = b.create<arith::IndexCastOp>(loc, b.getI64Type(), byteAddr);

    auto i1Ty = IntegerType::get(b.getContext(), 1);
    auto layout = StridedLayoutAttr::get(b.getContext(), /*offset=*/0, ArrayRef<int64_t>{1});
    auto i1MemrefTy = MemRefType::get({lanes}, i1Ty, layout, srcTy.getMemorySpace());
    return b.create<hivm::PointerCastOp>(loc, i1MemrefTy, addrI64).getResult();
}

static std::string buildUniqueVectorHelperName(ModuleOp module, int& nextVectorRegionId)
{
    std::string helperName;
    do {
        helperName = "vector_region_" + std::to_string(nextVectorRegionId++);
    } while (module.lookupSymbol<func::FuncOp>(helperName));
    return helperName;
}

enum class VectorBinaryKind
{
    Add,
    Sub,
    Mul,
    Div,
    Max,
    Min,
    And,
    Or,
    Xor,
    ShiftLeft,
    ShiftRight
};
enum class VectorRhsKind
{
    Vector,
    Scalar
};

static FailureOr<hivmave::CombiningKind> getAveReductionCombiningKind(::tla::ReduceOp reduceOp, Type elementType)
{
    auto kindAttr = reduceOp->getAttrOfType<StringAttr>("kind");
    if (!kindAttr)
        return reduceOp.emitError("tla.reduce requires string kind attribute"), failure();
    StringRef kind = kindAttr.getValue();
    if (kind == "add")
        return hivmave::CombiningKind::ADD;
    if (kind == "max") {
        if (auto intType = dyn_cast<IntegerType>(elementType))
            return intType.getSignedness() == IntegerType::Unsigned ? hivmave::CombiningKind::UMAX :
                                                                      hivmave::CombiningKind::MAX;
        if (isa<FloatType>(elementType))
            return hivmave::CombiningKind::MAX;
    }
    if (kind == "min") {
        if (auto intType = dyn_cast<IntegerType>(elementType))
            return intType.getSignedness() == IntegerType::Unsigned ? hivmave::CombiningKind::UMIN :
                                                                      hivmave::CombiningKind::MIN;
        if (isa<FloatType>(elementType))
            return hivmave::CombiningKind::MIN;
    }
    return reduceOp.emitError() << "tla.reduce supports only add, max, and min reductions, got \"" << kind << "\"",
           failure();
}

static bool isSupportedVectorReductionElementType(Type elementType)
{
    if (isa<Float16Type, Float32Type>(elementType))
        return true;
    auto intType = dyn_cast<IntegerType>(elementType);
    if (!intType)
        return false;
    switch (intType.getWidth()) {
        case 16:
        case 32:
            return true;
        default:
            return false;
    }
}

// The tla.tensor_descs a region uses but does not contain.
//
// tla-lower-tensor-desc is the sole descriptor producer, so this is a
// TensorDesc query and nothing here re-walks a tile_view / make_tensor producer
// chain. Both consumers -- the scalar capture in
// collectVectorHelperScalarOperands and the materialization in buildHelperFunc
// -- read the same list, so the two cannot drift apart.
static void collectExternalTensorDescs(::tla::VecFuncOp vecFuncOp, SmallVectorImpl<::tla::TensorDescOp>& descs)
{
    vecFuncOp.walk([&](Operation* op) {
        for (Value operand : op->getOperands()) {
            auto desc = operand.getDefiningOp<::tla::TensorDescOp>();
            if (desc && !vecFuncOp->isProperAncestor(desc.getOperation()) && !llvm::is_contained(descs, desc))
                descs.push_back(desc);
        }
    });
}

// A store destination tile is directly a tla.tensor_desc result after
// tla-lower-tensor-desc (the sole descriptor producer). Only look one level;
// the caller diagnoses when the value is not a descriptor result.
static ::tla::TensorDescOp findTensorDescProducer(Value tensorValue)
{
    return tensorValue ? tensorValue.getDefiningOp<::tla::TensorDescOp>() : nullptr;
}

static LogicalResult validateVectorReduction(::tla::ReduceOp reduceOp, Type elementType)
{
    if (!isSupportedVectorReductionElementType(elementType))
        return reduceOp.emitError() << "tla.reduce unsupported reduction element type " << elementType;
    return success();
}

enum class VectorUnaryKind
{
    Exp,
    Log,
    Sqrt,
    Abs,
    Neg,
    Not
};

struct TlaUnaryOperands {
    Value operand;
    Value mask;
};

struct VectorUnaryInfo {
    VectorUnaryKind kind;
    StringRef name;
    TlaUnaryOperands operands;
};

template <typename OpTy>
static TlaUnaryOperands getTlaUnaryOperands(OpTy op)
{
    return TlaUnaryOperands{op.getOperand(), op.getMask()};
}

static std::optional<VectorUnaryInfo> getVectorUnaryInfo(Operation* op)
{
    if (!op)
        return std::nullopt;
    if (auto o = dyn_cast<::tla::ExpOp>(op))
        return VectorUnaryInfo{VectorUnaryKind::Exp, "exp", getTlaUnaryOperands(o)};
    if (auto o = dyn_cast<::tla::LogOp>(op))
        return VectorUnaryInfo{VectorUnaryKind::Log, "log", getTlaUnaryOperands(o)};
    if (auto o = dyn_cast<::tla::SqrtOp>(op))
        return VectorUnaryInfo{VectorUnaryKind::Sqrt, "sqrt", getTlaUnaryOperands(o)};
    if (auto o = dyn_cast<::tla::AbsOp>(op))
        return VectorUnaryInfo{VectorUnaryKind::Abs, "abs", getTlaUnaryOperands(o)};
    if (auto o = dyn_cast<::tla::NegOp>(op))
        return VectorUnaryInfo{VectorUnaryKind::Neg, "neg", getTlaUnaryOperands(o)};
    if (auto o = dyn_cast<::tla::BitwiseNotOp>(op))
        return VectorUnaryInfo{VectorUnaryKind::Not, "bitwise_not", getTlaUnaryOperands(o)};
    return std::nullopt;
}

static LogicalResult validateVectorUnaryElementType(Operation* op, VectorUnaryInfo info, Type elementType)
{
    switch (info.kind) {
        case VectorUnaryKind::Exp:
        case VectorUnaryKind::Log:
        case VectorUnaryKind::Sqrt:
            if (!isa<FloatType>(elementType))
                return op->emitError() << "tla." << info.name << " requires floating-point element type, got "
                                       << elementType;
            if (isa<BFloat16Type>(elementType))
                return op->emitError() << "tla." << info.name << " does not support bf16 element type yet";
            return success();
        case VectorUnaryKind::Abs:
        case VectorUnaryKind::Neg:
        case VectorUnaryKind::Not:
            if (auto floatType = dyn_cast<FloatType>(elementType)) {
                if (isa<BFloat16Type>(floatType))
                    return op->emitError() << "tla." << info.name << " does not support bf16 element type yet";
                if (floatType.isF16() || floatType.isF32())
                    return success();
                return op->emitError() << "tla." << info.name
                                       << " requires f16 or f32 floating-point element type, got " << elementType;
            }
            if (auto intType = dyn_cast<IntegerType>(elementType)) {
                unsigned width = intType.getWidth();
                if (width == 8 || width == 16 || width == 32)
                    return success();
                return op->emitError() << "tla." << info.name << " requires i8, i16, or i32 element type, got "
                                       << elementType;
            }
            return op->emitError() << "tla." << info.name << " requires f16/f32 or i8/i16/i32 element type, got "
                                   << elementType;
    }
    return failure();
}

// The lhs/rhs/mask operands of vector binary ops (mask may be null).
struct TlaBinaryOperands {
    Value lhs;
    Value rhs;
    Value mask;
};

static TlaBinaryOperands getTlaBinaryOperands(Operation* op)
{
    TlaBinaryOperands r{};
    if (auto o = dyn_cast<::tla::AddOp>(op)) {
        r.lhs = o.getLhs();
        r.rhs = o.getRhs();
        r.mask = o.getMask();
    } else if (auto o = dyn_cast<::tla::SubOp>(op)) {
        r.lhs = o.getLhs();
        r.rhs = o.getRhs();
        r.mask = o.getMask();
    } else if (auto o = dyn_cast<::tla::MulOp>(op)) {
        r.lhs = o.getLhs();
        r.rhs = o.getRhs();
        r.mask = o.getMask();
    } else if (auto o = dyn_cast<::tla::DivOp>(op)) {
        r.lhs = o.getLhs();
        r.rhs = o.getRhs();
        r.mask = o.getMask();
    } else if (auto o = dyn_cast<::tla::MaxOp>(op)) {
        r.lhs = o.getLhs();
        r.rhs = o.getRhs();
        r.mask = o.getMask();
    } else if (auto o = dyn_cast<::tla::MinOp>(op)) {
        r.lhs = o.getLhs();
        r.rhs = o.getRhs();
        r.mask = o.getMask();
    } else if (auto o = dyn_cast<::tla::AddsOp>(op)) {
        r.lhs = o.getLhs();
        r.rhs = o.getRhs();
        r.mask = o.getMask();
    } else if (auto o = dyn_cast<::tla::SubsOp>(op)) {
        r.lhs = o.getLhs();
        r.rhs = o.getRhs();
        r.mask = o.getMask();
    } else if (auto o = dyn_cast<::tla::MulsOp>(op)) {
        r.lhs = o.getLhs();
        r.rhs = o.getRhs();
        r.mask = o.getMask();
    } else if (auto o = dyn_cast<::tla::MaxsOp>(op)) {
        r.lhs = o.getLhs();
        r.rhs = o.getRhs();
        r.mask = o.getMask();
    } else if (auto o = dyn_cast<::tla::MinsOp>(op)) {
        r.lhs = o.getLhs();
        r.rhs = o.getRhs();
        r.mask = o.getMask();
    } else if (auto o = dyn_cast<::tla::DivsOp>(op)) {
        r.lhs = o.getLhs();
        r.rhs = o.getRhs();
        r.mask = o.getMask();
    } else if (auto o = dyn_cast<::tla::ShiftLeftOp>(op)) {
        r.lhs = o.getLhs();
        r.rhs = o.getRhs();
        r.mask = o.getMask();
    } else if (auto o = dyn_cast<::tla::ShiftRightOp>(op)) {
        r.lhs = o.getLhs();
        r.rhs = o.getRhs();
        r.mask = o.getMask();
    } else if (auto o = dyn_cast<::tla::ShiftLeftsOp>(op)) {
        r.lhs = o.getLhs();
        r.rhs = o.getRhs();
        r.mask = o.getMask();
    } else if (auto o = dyn_cast<::tla::ShiftRightsOp>(op)) {
        r.lhs = o.getLhs();
        r.rhs = o.getRhs();
        r.mask = o.getMask();
    } else if (auto o = dyn_cast<::tla::BitwiseAndOp>(op)) {
        r.lhs = o.getLhs();
        r.rhs = o.getRhs();
        r.mask = o.getMask();
    } else if (auto o = dyn_cast<::tla::BitwiseOrOp>(op)) {
        r.lhs = o.getLhs();
        r.rhs = o.getRhs();
        r.mask = o.getMask();
    } else if (auto o = dyn_cast<::tla::BitwiseXorOp>(op)) {
        r.lhs = o.getLhs();
        r.rhs = o.getRhs();
        r.mask = o.getMask();
    }
    return r;
}

struct VectorOpInfo {
    VectorBinaryKind kind;
    VectorRhsKind rhsKind;
    StringRef mnemonic;
    TlaBinaryOperands operands;
};

struct AnyVectorOperationInfo {
    std::optional<VectorOpInfo> binary;
    std::optional<VectorUnaryInfo> unary;
};

static std::optional<VectorOpInfo> getVectorBinaryInfo(Operation* op)
{
    if (!op)
        return std::nullopt;
    if (isa<::tla::AddOp>(op))
        return VectorOpInfo{VectorBinaryKind::Add, VectorRhsKind::Vector, "add", getTlaBinaryOperands(op)};
    if (isa<::tla::SubOp>(op))
        return VectorOpInfo{VectorBinaryKind::Sub, VectorRhsKind::Vector, "sub", getTlaBinaryOperands(op)};
    if (isa<::tla::MulOp>(op))
        return VectorOpInfo{VectorBinaryKind::Mul, VectorRhsKind::Vector, "mul", getTlaBinaryOperands(op)};
    if (isa<::tla::DivOp>(op))
        return VectorOpInfo{VectorBinaryKind::Div, VectorRhsKind::Vector, "div", getTlaBinaryOperands(op)};
    if (isa<::tla::MaxOp>(op))
        return VectorOpInfo{VectorBinaryKind::Max, VectorRhsKind::Vector, "max", getTlaBinaryOperands(op)};
    if (isa<::tla::MinOp>(op))
        return VectorOpInfo{VectorBinaryKind::Min, VectorRhsKind::Vector, "min", getTlaBinaryOperands(op)};
    if (isa<::tla::BitwiseAndOp>(op))
        return VectorOpInfo{VectorBinaryKind::And, VectorRhsKind::Vector, "bitwise_and", getTlaBinaryOperands(op)};
    if (isa<::tla::BitwiseOrOp>(op))
        return VectorOpInfo{VectorBinaryKind::Or, VectorRhsKind::Vector, "bitwise_or", getTlaBinaryOperands(op)};
    if (isa<::tla::BitwiseXorOp>(op))
        return VectorOpInfo{VectorBinaryKind::Xor, VectorRhsKind::Vector, "bitwise_xor", getTlaBinaryOperands(op)};
    if (isa<::tla::ShiftLeftOp>(op))
        return VectorOpInfo{VectorBinaryKind::ShiftLeft, VectorRhsKind::Vector, "shift_left", getTlaBinaryOperands(op)};
    if (isa<::tla::ShiftRightOp>(op))
        return VectorOpInfo{
            VectorBinaryKind::ShiftRight, VectorRhsKind::Vector, "shift_right", getTlaBinaryOperands(op)};
    return std::nullopt;
}

static std::optional<VectorOpInfo> getVectorScalarBinaryInfo(Operation* op)
{
    if (!op)
        return std::nullopt;
    if (isa<::tla::AddsOp>(op))
        return VectorOpInfo{VectorBinaryKind::Add, VectorRhsKind::Scalar, "adds", getTlaBinaryOperands(op)};
    if (isa<::tla::SubsOp>(op))
        return VectorOpInfo{VectorBinaryKind::Sub, VectorRhsKind::Scalar, "subs", getTlaBinaryOperands(op)};
    if (isa<::tla::MulsOp>(op))
        return VectorOpInfo{VectorBinaryKind::Mul, VectorRhsKind::Scalar, "muls", getTlaBinaryOperands(op)};
    if (isa<::tla::MaxsOp>(op))
        return VectorOpInfo{VectorBinaryKind::Max, VectorRhsKind::Scalar, "maxs", getTlaBinaryOperands(op)};
    if (isa<::tla::MinsOp>(op))
        return VectorOpInfo{VectorBinaryKind::Min, VectorRhsKind::Scalar, "mins", getTlaBinaryOperands(op)};
    if (isa<::tla::DivsOp>(op))
        return VectorOpInfo{VectorBinaryKind::Div, VectorRhsKind::Scalar, "divs", getTlaBinaryOperands(op)};
    if (isa<::tla::ShiftLeftsOp>(op))
        return VectorOpInfo{
            VectorBinaryKind::ShiftLeft, VectorRhsKind::Scalar, "shift_lefts", getTlaBinaryOperands(op)};
    if (isa<::tla::ShiftRightsOp>(op))
        return VectorOpInfo{
            VectorBinaryKind::ShiftRight, VectorRhsKind::Scalar, "shift_rights", getTlaBinaryOperands(op)};
    return std::nullopt;
}

static std::optional<AnyVectorOperationInfo> getAnyVectorOperationInfo(Operation* op)
{
    if (auto info = getVectorBinaryInfo(op))
        return AnyVectorOperationInfo{*info, std::nullopt};
    if (auto info = getVectorScalarBinaryInfo(op))
        return AnyVectorOperationInfo{*info, std::nullopt};
    if (auto info = getVectorUnaryInfo(op))
        return AnyVectorOperationInfo{std::nullopt, *info};
    return std::nullopt;
}

// The mask-register width (b8/b16/b32) matching the element type.
static hivmave::MaskWidth maskWidthForElement(Type elementType)
{
    unsigned bits = elementType.getIntOrFloatBitWidth();
    if (bits <= 8)
        return hivmave::MaskWidth::B8;
    if (bits <= 16)
        return hivmave::MaskWidth::B16;
    return hivmave::MaskWidth::B32;
}

static std::optional<hivmave::CmpType> mapCmpMode(StringRef mode)
{
    return llvm::StringSwitch<std::optional<hivmave::CmpType>>(mode)
        .Case("lt", hivmave::CmpType::LT)
        .Case("le", hivmave::CmpType::LE)
        .Case("gt", hivmave::CmpType::GT)
        .Case("ge", hivmave::CmpType::GE)
        .Case("eq", hivmave::CmpType::EQ)
        .Case("ne", hivmave::CmpType::NE)
        .Default(std::nullopt);
}

// True for the tla ops that produce a vector compute result inside a vec.func
// region: element-wise binary/unary ops, bitwise ops, where/select,
// reductions, and gather.
static bool isVectorComputeOp(Operation* op)
{
    return getAnyVectorOperationInfo(op).has_value() || isa_and_nonnull<::tla::CmpOp>(op) ||
           isa_and_nonnull<::tla::WhereOp>(op) || isa_and_nonnull<::tla::SqueezeOp>(op) ||
           isa_and_nonnull<::tla::ReduceOp>(op) || isa_and_nonnull<::tla::GatherOp>(op) ||
           isa_and_nonnull<::tla::CastOp>(op) || isa_and_nonnull<::tla::InterleaveOp>(op) ||
           isa_and_nonnull<::tla::DeInterleaveOp>(op);
}

static std::string getSqueezeLibraryCallName(Type elementType)
{
    if (elementType.isF32())
        return "vsqueeze_float";
    if (elementType.isF16())
        return "vsqueeze_half";
    if (auto intType = dyn_cast<IntegerType>(elementType))
        if (intType.getWidth() == 32)
            return "vsqueeze_int32_t";
    return {};
}

static func::FuncOp getOrCreateSqueezeLibraryCall(
    ModuleOp module, Location loc, VectorType vecType, VectorType pregType, StringRef calleeName)
{
    if (auto existing = module.lookupSymbol<func::FuncOp>(calleeName))
        return existing;
    OpBuilder moduleBuilder(module.getBodyRegion());
    auto fnType = FunctionType::get(module.getContext(), {vecType, pregType}, {vecType});
    auto callee = moduleBuilder.create<func::FuncOp>(loc, calleeName, fnType);
    callee.setPrivate();
    callee->setAttr("llvm.emit_c_interface", UnitAttr::get(module.getContext()));
    return callee;
}

static std::string getStoreWithStrideLibraryCallName(Type elementType)
{
    if (elementType.isF32()) {
        return "store_with_stride_float";
    } else if (elementType.isF16()) {
        return "store_with_stride_half";
    } else if (elementType.isBF16()) {
        return "store_with_stride_bf16";
    } else if (isa<Float8E4M3FNType>(elementType)) {
        return "store_with_stride_fp8_e4m3fn";
    } else if (isa<Float8E5M2Type>(elementType)) {
        return "store_with_stride_fp8_e5m2";
    } else if (elementType.isInteger(8) && !elementType.isUnsignedInteger()) {
        return "store_with_stride_int8";
    }
    return {};
}

// The 8-bit block load/store stubs take the register as int8_t, because the AIV
// backend has no scalar semantics for fp8 and so no VectorReg<fp8_*>. The move
// itself is width-and-stride only, so bitcasting the register costs nothing.
static bool blockStrideUsesByteRegister(Type elementType)
{
    return isa<Float8E4M3FNType, Float8E5M2Type>(elementType);
}

static VectorType byteRegisterTypeFor(VectorType vecType)
{
    return VectorType::get(vecType.getShape(), IntegerType::get(vecType.getContext(), 8));
}

static void annotateStoreWithStrideLibraryCall(func::FuncOp callee)
{
    MLIRContext* ctx = callee.getContext();
    callee.setPrivate();
    callee->setAttr("llvm.emit_c_interface", UnitAttr::get(ctx));
    callee->setAttr(hivm::TFuncCoreTypeAttr::name, hivm::TFuncCoreTypeAttr::get(ctx, hivm::TFuncCoreType::AIV));
}

// Strided block load (vsldb) library call names, mirroring the AVE bc stubs in
// bc/Vector/load.cpp: one symbol per element type.
static std::string getLoadWithStrideLibraryCallName(Type elementType)
{
    if (elementType.isF32()) {
        return "load_with_stride_float";
    } else if (elementType.isF16()) {
        return "load_with_stride_half";
    } else if (elementType.isBF16()) {
        return "load_with_stride_bf16";
    } else if (elementType.isInteger(32)) {
        return elementType.isUnsignedInteger() ? "load_with_stride_uint32" : "load_with_stride_int32";
    } else if (elementType.isInteger(16)) {
        return elementType.isUnsignedInteger() ? "load_with_stride_uint16" : "load_with_stride_int16";
    } else if (isa<Float8E4M3FNType>(elementType)) {
        return "load_with_stride_fp8_e4m3fn";
    } else if (isa<Float8E5M2Type>(elementType)) {
        return "load_with_stride_fp8_e5m2";
    } else if (elementType.isInteger(8) && !elementType.isUnsignedInteger()) {
        return "load_with_stride_int8";
    }
    return {};
}

static func::FuncOp getOrCreateStoreWithStrideLibraryCall(
    ModuleOp module, Location loc, Type vecType, Type memRefType, StringRef calleeName)
{
    auto ctx = module.getContext();
    if (auto existing = module.lookupSymbol<func::FuncOp>(calleeName)) {
        annotateStoreWithStrideLibraryCall(existing);
        return existing;
    }
    OpBuilder moduleBuilder(module.getBodyRegion());
    auto fnType = FunctionType::get(
        ctx, {vecType, memRefType, IntegerType::get(ctx, 32), VectorType::get({256}, IntegerType::get(ctx, 1))}, {});
    auto callee = moduleBuilder.create<func::FuncOp>(loc, calleeName, fnType);
    annotateStoreWithStrideLibraryCall(callee);
    return callee;
}

// Strided block load (vsldb) library call: (memref, blockStride, repeatStride,
// preg) -> vecType. Annotated the same way as the store-with-stride call.
static func::FuncOp getOrCreateLoadWithStrideLibraryCall(
    ModuleOp module, Location loc, Type vecType, Type memRefType, StringRef calleeName)
{
    auto ctx = module.getContext();
    if (auto existing = module.lookupSymbol<func::FuncOp>(calleeName)) {
        annotateStoreWithStrideLibraryCall(existing);
        return existing;
    }
    OpBuilder moduleBuilder(module.getBodyRegion());
    auto fnType = FunctionType::get(
        ctx,
        {memRefType, IntegerType::get(ctx, 32), IntegerType::get(ctx, 32),
         VectorType::get({256}, IntegerType::get(ctx, 1))},
        {vecType});
    auto callee = moduleBuilder.create<func::FuncOp>(loc, calleeName, fnType);
    annotateStoreWithStrideLibraryCall(callee);
    return callee;
}

static Value castMaskToPregType(OpBuilder& b, Location loc, Value mask, VectorType pregVecType)
{
    if (mask.getType() == pregVecType)
        return mask;
    return b.create<UnrealizedConversionCastOp>(loc, pregVecType, mask).getResult(0);
}

static VectorType fullPregVecType(MLIRContext* ctx)
{
    return VectorType::get({256}, IntegerType::get(ctx, 1));
}

static hivmave::MaskWidthAttr maskWidthAttrForElement(OpBuilder& b, Type elementType)
{
    return hivmave::MaskWidthAttr::get(b.getContext(), maskWidthForElement(elementType));
}

// The semantic width of a MaskSSA is carried by !tla.mask<N>, independently
// of its lowered predicate-register container. A carrier crossing SCF uses the
// backend-native vector<256xi1> container, but !tla.mask<64> must still select
// B32 rather than being misclassified as B8 from that container width.
static hivmave::MaskWidth maskWidthForMaskType(::tla::MaskSSAType maskType)
{
    int64_t lanes = maskType.getPhysicalLanes();
    if (lanes <= 0)
        return hivmave::MaskWidth::B32;
    int64_t bytesPerLane = 256 / lanes;
    if (bytesPerLane <= 1)
        return hivmave::MaskWidth::B8;
    if (bytesPerLane <= 2)
        return hivmave::MaskWidth::B16;
    return hivmave::MaskWidth::B32;
}

static hivmave::MaskWidthAttr maskWidthAttrForMaskType(OpBuilder& b, ::tla::MaskSSAType maskType)
{
    return hivmave::MaskWidthAttr::get(b.getContext(), maskWidthForMaskType(maskType));
}

// Build the AVE vector op for a tla binary op. The mask controls active lanes.
// For div the signedness is carried as the TypeFn cast attribute (cast_unsigned
// for unsigned integer element types, cast_signed otherwise).
static Value createVectorBinaryResult(
    OpBuilder& b, Location loc, VectorBinaryKind kind, Type tlaOperandType, Type elementType, VectorType vecType,
    Value lhs, Value rhs, Value mask)
{
    switch (kind) {
        case VectorBinaryKind::Add:
            return b.create<hivmave::VFAddOp>(loc, vecType, lhs, rhs, mask, Value()).getResult();
        case VectorBinaryKind::Sub:
            return b.create<hivmave::VFSubOp>(loc, vecType, lhs, rhs, mask, Value()).getResult();
        case VectorBinaryKind::Mul:
            return b.create<hivmave::VFMulOp>(loc, vecType, lhs, rhs, mask, Value()).getResult();
        case VectorBinaryKind::Div: {
            auto cast = hivm::TypeFn::cast_signed;
            if (auto intType = dyn_cast<IntegerType>(elementType))
                if (intType.getSignedness() == IntegerType::Unsigned)
                    cast = hivm::TypeFn::cast_unsigned;
            return b
                .create<hivmave::VFDivOp>(
                    loc, vecType, lhs, rhs, mask, hivm::TypeFnAttr::get(b.getContext(), cast), Value())
                .getResult();
        }
        case VectorBinaryKind::Max:
            return b.create<hivmave::VFMaxOp>(loc, vecType, lhs, rhs, mask, Value()).getResult();
        case VectorBinaryKind::Min:
            return b.create<hivmave::VFMinOp>(loc, vecType, lhs, rhs, mask, Value()).getResult();
        case VectorBinaryKind::And:
            if (isa<::tla::VectorSSAType>(tlaOperandType))
                return b.create<hivmave::VFAndOp>(loc, vecType, lhs, rhs, mask, Value()).getResult();
            if (isa<::tla::MaskSSAType>(tlaOperandType))
                return b
                    .create<hivmave::PregAndOp>(
                        loc, vecType, maskWidthAttrForMaskType(b, cast<::tla::MaskSSAType>(tlaOperandType)), lhs, rhs,
                        mask)
                    .getRes();
            return nullptr;
        case VectorBinaryKind::Or:
            if (isa<::tla::VectorSSAType>(tlaOperandType))
                return b.create<hivmave::VFOrOp>(loc, vecType, lhs, rhs, mask, Value()).getResult();
            if (isa<::tla::MaskSSAType>(tlaOperandType))
                return b
                    .create<hivmave::PregOrOp>(
                        loc, vecType, maskWidthAttrForMaskType(b, cast<::tla::MaskSSAType>(tlaOperandType)), lhs, rhs,
                        mask)
                    .getRes();
            return nullptr;
        case VectorBinaryKind::Xor:
            if (isa<::tla::VectorSSAType>(tlaOperandType))
                return b.create<hivmave::VFXorOp>(loc, vecType, lhs, rhs, mask, Value()).getResult();
            if (isa<::tla::MaskSSAType>(tlaOperandType))
                return b
                    .create<hivmave::PregXorOp>(
                        loc, vecType, maskWidthAttrForMaskType(b, cast<::tla::MaskSSAType>(tlaOperandType)), lhs, rhs,
                        mask)
                    .getRes();
            return nullptr;
        case VectorBinaryKind::ShiftLeft:
        case VectorBinaryKind::ShiftRight: {
            // The verifier restricts shift sources to signed (signless)
            // i8/i16/i32 and the shift vector to the same signless width.
            // Hence the signed intrinsics are the only valid lowering.
            if (!dyn_cast<IntegerType>(elementType))
                return nullptr;
            if (kind == VectorBinaryKind::ShiftLeft)
                return b.create<hivm_regbaseintrins::VshlSXInstrOp>(loc, vecType, lhs, rhs, mask).getRes();
            return b.create<hivm_regbaseintrins::VshrSXInstrOp>(loc, vecType, lhs, rhs, mask).getRes();
        }
    }
    return nullptr;
}

static int64_t maskElementBitWidthForLanes(int64_t lanes)
{
    if (lanes >= 256)
        return 8;
    if (lanes >= 128)
        return 16;
    return 32;
}

// AVE represents a semantic vector<Nxi1> predicate as a hardware
// vector<256xi1>. When those types differ, annotate the AVE producer with the
// semantic element width so HIVMAVE lowering still selects pge/plt.b8/b16/b32
// from N instead of the full container width.
static void annotateFullPregWidth(OpBuilder& b, Operation* op, VectorType resultType, int64_t semanticLanes)
{
    if (resultType.getNumElements() == semanticLanes)
        return;
    op->setAttr(mlir::utils::elementAlignmentBitWidth, b.getI32IntegerAttr(maskElementBitWidthForLanes(semanticLanes)));
}

static Value createPredicatePge(
    OpBuilder& b, Location loc, VectorType resultType, int64_t semanticLanes, hivmave::PgePattern pattern)
{
    auto pge = createAvePgeMask(b, loc, resultType, pattern);
    annotateFullPregWidth(b, pge, resultType, semanticLanes);
    return pge.getRes();
}

static hivmave::VFPltOp createPredicatePlt(
    OpBuilder& b, Location loc, VectorType resultType, int64_t semanticLanes, Value trueShape)
{
    auto plt = createAvePltMask(b, loc, resultType, trueShape);
    annotateFullPregWidth(b, plt, resultType, semanticLanes);
    return plt;
}

// Reuse auto-generated full predicates at the vector-helper entry. A generated
// predicate has no data dependency, so placing it here makes it dominate every
// loop and branch in the helper. Explicit tla.create_mask stays at its source
// location; this cache applies only to omitted masks.
class AutoFullPredicateCache {
public:
    explicit AutoFullPredicateCache(Block* entry) : entry(entry)
    {}

    Value get(Location loc, int64_t semanticLanes)
    {
        if (Value mask = masks.lookup(semanticLanes))
            return mask;
        OpBuilder entryBuilder = OpBuilder::atBlockBegin(entry);
        // Keep the cached predicate in the hardware register-container type.
        // A semantic-width cache value would be materialized as a conversion
        // to vector<Nxi1> at the entry and then converted back to the physical
        // predicate for every vector op. Explicit tla.create_mask lowers as the
        // physical predicate directly, so the hidden default must do the same.
        VectorType maskType = fullPregVecType(entry->getParent()->getContext());
        Value mask = createPredicatePge(entryBuilder, loc, maskType, semanticLanes, hivmave::PgePattern::ALL);
        masks[semanticLanes] = mask;
        return mask;
    }

private:
    Block* entry;
    DenseMap<uint64_t, Value> masks;
};

// An all-lanes-active predicate for a data or MaskSSA vector. MaskSSA keeps its
// semantic lane count in tlaOperandType even when its mapped value is the full
// predicate-register container.
static Value allTrueMaskFor(AutoFullPredicateCache& cache, Location loc, VectorType vecType, Type tlaOperandType)
{
    int64_t semanticLanes = vecType.getNumElements();
    if (auto maskType = dyn_cast<::tla::MaskSSAType>(tlaOperandType))
        semanticLanes = maskType.getPhysicalLanes();
    return cache.get(loc, semanticLanes);
}

// Map the tla.cast round mode onto the HIVM round_mode attribute.
static hivm::RoundModeAttr mapCastRoundMode(OpBuilder& b, ::RoundMode mode)
{
    hivm::RoundMode hv = hivm::RoundMode::ROUND;
    switch (mode) {
        case ::RoundMode::cast_round:
            hv = hivm::RoundMode::ROUND;
            break;
        case ::RoundMode::cast_floor:
            hv = hivm::RoundMode::FLOOR;
            break;
        case ::RoundMode::cast_ceil:
            hv = hivm::RoundMode::CEIL;
            break;
        case ::RoundMode::cast_trunc:
            hv = hivm::RoundMode::TRUNC;
            break;
    }
    return hivm::RoundModeAttr::get(b.getContext(), hv);
}

// Map the tla.cast register layout onto the AVE VCVT part (even/odd) attribute.
static hivmave::VCVT_PartTypeAttr mapCastPart(OpBuilder& b, ::RegSlot layout)
{
    auto part = layout == ::RegSlot::one ? hivmave::VCVT_PartType::PART_ODD : hivmave::VCVT_PartType::PART_EVEN;
    return hivmave::VCVT_PartTypeAttr::get(b.getContext(), part);
}

// Map the tla.cast register layout onto the AVE pack pattern (pp0..pp3) used by
// 4x-width int casts (i32<->i8). reg_slot zero/one/two/three -> pp0/pp1/pp2/pp3.
static hivmave::VCVT_PPTypeAttr mapCastPP(OpBuilder& b, ::RegSlot layout)
{
    hivmave::VCVT_PPType pp;
    switch (layout) {
        case ::RegSlot::one:
            pp = hivmave::VCVT_PPType::PP1;
            break;
        case ::RegSlot::two:
            pp = hivmave::VCVT_PPType::PP2;
            break;
        case ::RegSlot::three:
            pp = hivmave::VCVT_PPType::PP3;
            break;
        case ::RegSlot::zero:
        default:
            pp = hivmave::VCVT_PPType::PP0;
            break;
    }
    return hivmave::VCVT_PPTypeAttr::get(b.getContext(), pp);
}

// The two 8-bit floats AVE vtruncf / vextf actually accept -- not every 8-bit
// float MLIR knows. e8m0 has no AVE conversion path and must not slip in on
// width alone.
static bool isAveFloat8Type(Type t)
{
    return isa<Float8E4M3FNType, Float8E5M2Type>(t);
}

// HIVMAVE's float conversion ops model only the 2x even/odd selector. FP8
// <-> f32 is the one 4x float conversion: its hardware intrinsics take the
// pp0..pp3 immediate directly. Materialize those intrinsics here rather than
// silently folding reg_slot TWO/THREE into PART_EVEN in the generic AVE op.
static Value createFp8ToF32Result(
    OpBuilder& b, Location loc, VectorType dstVecType, Value src, Value mask, hivmave::VCVT_PPTypeAttr pp)
{
    Value preg = castMaskToPregType(b, loc, mask, fullPregVecType(b.getContext()));
    Value part = b.create<arith::ConstantIntOp>(loc, static_cast<int64_t>(pp.getValue()), 32);
    Type elemType = cast<VectorType>(src.getType()).getElementType();
    if (isa<Float8E4M3FNType>(elemType))
        return b.create<hivm_regbaseintrins::VcvtffF8E4M32F32InstrOp>(loc, dstVecType, src, preg, part).getResult();
    return b.create<hivm_regbaseintrins::VcvtffF8E5M22F32InstrOp>(loc, dstVecType, src, preg, part).getResult();
}

static Value createF32ToFp8Result(
    OpBuilder& b, Location loc, VectorType dstVecType, Value src, Value mask, hivm::RoundModeAttr rnd, BoolAttr sat,
    hivmave::VCVT_PPTypeAttr pp)
{
    Value preg = castMaskToPregType(b, loc, mask, fullPregVecType(b.getContext()));
    Value round = b.create<arith::ConstantIntOp>(loc, static_cast<int64_t>(rnd.getValue()), 32);
    Value saturation = b.create<arith::ConstantIntOp>(loc, sat.getValue() ? 1 : 0, 32);
    Value part = b.create<arith::ConstantIntOp>(loc, static_cast<int64_t>(pp.getValue()), 32);
    Type elemType = dstVecType.getElementType();
    if (isa<Float8E4M3FNType>(elemType))
        return b
            .create<hivm_regbaseintrins::VcvtffF322F8E4M3InstrOp>(loc, dstVecType, src, preg, round, saturation, part)
            .getResult();
    return b.create<hivm_regbaseintrins::VcvtffF322F8E5M2InstrOp>(loc, dstVecType, src, preg, round, saturation, part)
        .getResult();
}

// E5M2 and f16 have the same sign and exponent layout (one sign bit and a
// five-bit, bias-15 exponent). Their two E5M2 mantissa bits become f16 bits
// 9:8, so widening the raw byte and shifting it left by eight is an exact
// conversion. Unlike the generic fp8 -> f32 -> f16 composition, this emits one
// 2x integer widen and one shift. The caller's even/odd slot selects which
// source elements become the result, identically to the composed route.
static Value createE5M2ToF16Result(
    OpBuilder& b, Location loc, VectorType srcVecType, VectorType dstVecType, Value src, Value mask,
    hivmave::VCVT_PartTypeAttr part)
{
    auto srcBitsType = VectorType::get(srcVecType.getShape(), b.getIntegerType(8));
    auto dstBitsType = VectorType::get(dstVecType.getShape(), b.getIntegerType(16));
    Value srcBits = b.create<mlir::vector::BitCastOp>(loc, srcBitsType, src).getResult();
    Value widened =
        b.create<hivmave::VFExtSIOp>(loc, dstBitsType, srcBits, mask, part, hivmave::VCVT_PPTypeAttr()).getResult();
    Value shift = b.create<arith::ConstantIntOp>(loc, 8, 16);
    Value sign = b.create<arith::ConstantOp>(loc, b.getI1Type(), b.getBoolAttr(false));
    Value shifted = b.create<hivmave::VFShlsOp>(loc, dstBitsType, widened, shift, mask, sign, Value()).getResult();
    return b.create<mlir::vector::BitCastOp>(loc, dstVecType, shifted).getResult();
}

// Element types the tla.cast lowering can emit AVE ops for: signed/signless
// integers i8/i16/i32/i64, floats f16/bf16/f32, and the fp8 formats
// f8E4M3FN/f8E5M2. Unsigned integers, i1 (bool) and f64 have no AVE cast path
// and are rejected (the front-end rejects them too; this guards hand-written /
// non-front-end IR).
static bool isSupportedCastElementType(Type t)
{
    if (auto f = dyn_cast<FloatType>(t))
        return f.getWidth() == 16 || f.getWidth() == 32 || isAveFloat8Type(t);
    if (auto i = dyn_cast<IntegerType>(t)) {
        if (i.isUnsigned() || i.getWidth() == 1) // unsigned / bool
            return false;
        unsigned w = i.getWidth();
        return w == 8 || w == 16 || w == 32 || w == 64;
    }
    return false;
}

// Build the AVE cast op for a tla.cast, dispatching by (src, dst) element kind.
// The trait supplies rounding, saturation and register layout; the mask (source
// width) predicates active lanes.
static FailureOr<Value> createVectorCastResult(
    OpBuilder& b, AutoFullPredicateCache& fullMasks, Location loc, VectorType srcVecType, VectorType dstVecType,
    ArrayRef<int32_t> trait, Value src, Value mask, bool hasUserMask)
{
    // trait codes: [0] reg_slot, [1] sat_mode, [2] round_mode.
    Type s = srcVecType.getElementType();
    Type d = dstVecType.getElementType();
    auto rnd = mapCastRoundMode(b, static_cast<::RoundMode>(trait[2]));
    BoolAttr sat = b.getBoolAttr(static_cast<::SatMode>(trait[1]) == ::SatMode::sat);
    auto part = mapCastPart(b, static_cast<::RegSlot>(trait[0]));
    auto pp = mapCastPP(b, static_cast<::RegSlot>(trait[0]));

    bool sFloat = isa<FloatType>(s);
    bool dFloat = isa<FloatType>(d);
    unsigned sb = s.getIntOrFloatBitWidth();
    unsigned db = d.getIntOrFloatBitWidth();

    // fp8 pairs with f32 and nothing else. The AVE op definitions advertise a
    // wider set -- vtruncf lists bf16/f16/f32 sources, vextf lists bf16/f16/f32
    // results -- but HIVMAVEToAVEIntrin only ever builds VcvtffF322F8E4M3 /
    // VcvtffF322F8E5M2 and VcvtffF8E4M32F32 / VcvtffF8E5M22F32. An f8 -> f16
    // widen hits llvm_unreachable there, and, more dangerously, an f16 -> f8
    // narrow does NOT: that path dispatches on the destination type alone, so
    // it silently emits the F32-source instruction against f16 lanes. Refuse
    // every non-f32 partner here rather than let that through. bf16/f16 reach
    // fp8 by widening to f32 first, which is exact.
    if (isAveFloat8Type(s) || isAveFloat8Type(d)) {
        Type partner = isAveFloat8Type(s) ? d : s;
        if (isAveFloat8Type(s) && isAveFloat8Type(d))
            return failure(); // no fp8 <-> fp8 re-encode instruction

        // Every fp8 route that is not a single instruction has the same mask
        // problem: its instructions run in different lane domains -- f32 or
        // f16 for the composed route, i8 then i16 for the e5m2 shortcut --
        // while the mask is written in the source element domain. An all-true
        // or contiguous mask is insensitive to the mismatch; a strided one is
        // not, and lanes are silently dropped.
        //
        // Fold the predicate into the data once, here, where the mask means
        // what it says, and let the instructions run all-true. A masked-off
        // element becomes zero and converts to zero, which is what the
        // hardware already yields for inactive lanes.
        //
        // Only the multi-instruction routes need this. The direct fp8 <-> f32
        // conversions are a single instruction that consumes the mask in the
        // source domain already, so gating them would emit an and, two
        // bitcasts and a pge that nothing reads.
        const bool needsGatedMask = (isa<Float8E5M2Type>(s) && d.isF16()) || !partner.isF32();
        Value gatedSrc = src;
        Value legMask = mask;
        if (hasUserMask && needsGatedMask) {
            auto srcBitsType = VectorType::get(srcVecType.getShape(), b.getIntegerType(sb));
            Value srcBits = b.create<mlir::vector::BitCastOp>(loc, srcBitsType, src).getResult();
            // and-with-self under the predicate: keeps active lanes, zeros the rest.
            Value gated = b.create<hivmave::VFAndOp>(loc, srcBitsType, srcBits, srcBits, mask, Value()).getResult();
            gatedSrc = b.create<mlir::vector::BitCastOp>(loc, srcVecType, gated).getResult();
            legMask = allTrueMaskFor(fullMasks, loc, srcVecType, srcVecType);
        }
        // E5M2 -> f16 has an exact two-instruction form (see below). It places
        // its result exactly where the composed route does -- both select the
        // source elements `part` names -- so it is simply the cheaper way to
        // spell the same cast, and is always preferred.
        if (isa<Float8E5M2Type>(s) && d.isF16())
            return createE5M2ToF16Result(b, loc, srcVecType, dstVecType, gatedSrc, legMask, part);
        if (!partner.isF32()) {
            // f16 and bf16 have no fp8 instruction, so expand the cast into the
            // two the ISA does have, through f32. The user writes one .to();
            // this is where it becomes two. Exact in the widening direction
            // (fp8 -> f32 loses nothing) and single-rounded in the other, since
            // only the f32 -> fp8 leg rounds.
            if (!partner.isF16() && !partner.isBF16())
                return failure();
            // Every vector register is 256 bytes, so the f32 intermediate holds
            // 2048 / 32 lanes whatever the endpoints are.
            auto f32Ty = b.getF32Type();
            auto midType = VectorType::get({static_cast<int64_t>(2048 / 32)}, f32Ty);
            // vextf and vtruncf each address one half of a packed register.
            // A single composed route therefore drops the other half. Convert
            // both halves and OR their disjoint output lanes together. vor is
            // an integer operation, so merge the raw destination bits before
            // restoring the requested float type. The truncation is the only
            // rounded step.
            auto even = hivmave::VCVT_PartTypeAttr::get(b.getContext(), hivmave::VCVT_PartType::PART_EVEN);
            auto odd = hivmave::VCVT_PartTypeAttr::get(b.getContext(), hivmave::VCVT_PartType::PART_ODD);
            // The f8 <-> f32 leg is a 4x conversion, so its selector is the
            // pack pattern pp0..pp3 (which byte of each 32-bit slot), NOT the
            // 2x even/odd part. The generic AVE ops carry only `part`, and
            // HIVMAVEToAVEIntrin forwards that attribute verbatim as the pp
            // immediate -- PART_EVEN/PART_ODD are numerically PP0/PP1 -- so
            // driving this leg with even/odd silently reads or writes only
            // byte positions 0 and 1 of every 4-byte group and never touches
            // 2 and 3. Half the register is dropped and the rest lands on the
            // wrong lanes. Build the leg from the pp intrinsics instead.
            //
            // Pick the pattern so the composed cast matches every other 2x
            // widen/narrow in this function, where reg_slot selects the even
            // or the odd elements: the f32 half contributes the high pp bit
            // and reg_slot the low one, i.e. pp = 2 * half + (reg_slot == one).
            // Widening then gives dst[j] = src[2j + slot], and narrowing gives
            // dst[2j + slot] = src[j], which is also what the e5m2 -> f16
            // integer-widen path above produces.
            unsigned slotBit = (static_cast<::RegSlot>(trait[0]) == ::RegSlot::one) ? 1 : 0;
            auto ppForHalf = [&](unsigned half) {
                return hivmave::VCVT_PPTypeAttr::get(
                    b.getContext(), static_cast<hivmave::VCVT_PPType>(2 * half + slotBit));
            };
            Value dstEven;
            Value dstOdd;
            if (isAveFloat8Type(s)) {
                // fp8 -> f16/bf16. Widen the two source groups the destination
                // can hold, then narrow each onto the dst even / odd lanes.
                Value midEven = createFp8ToF32Result(b, loc, midType, gatedSrc, legMask, ppForHalf(0));
                Value midOdd = createFp8ToF32Result(b, loc, midType, gatedSrc, legMask, ppForHalf(1));
                dstEven = b.create<hivmave::VFTruncFOp>(loc, dstVecType, midEven, legMask, rnd, sat, even).getResult();
                dstOdd = b.create<hivmave::VFTruncFOp>(loc, dstVecType, midOdd, legMask, rnd, sat, odd).getResult();
            } else {
                // f16/bf16 -> fp8. Widen the source even / odd lanes, then
                // narrow each into its own byte position of the destination.
                Value midEven = b.create<hivmave::VFExtFOp>(loc, midType, gatedSrc, legMask, even).getResult();
                Value midOdd = b.create<hivmave::VFExtFOp>(loc, midType, gatedSrc, legMask, odd).getResult();
                dstEven = createF32ToFp8Result(b, loc, dstVecType, midEven, legMask, rnd, sat, ppForHalf(0));
                dstOdd = createF32ToFp8Result(b, loc, dstVecType, midOdd, legMask, rnd, sat, ppForHalf(1));
            }
            auto dstBitsType = VectorType::get(dstVecType.getShape(), b.getIntegerType(d.getIntOrFloatBitWidth()));
            Value dstEvenBits = b.create<mlir::vector::BitCastOp>(loc, dstBitsType, dstEven).getResult();
            Value dstOddBits = b.create<mlir::vector::BitCastOp>(loc, dstBitsType, dstOdd).getResult();
            Value dstMask = allTrueMaskFor(fullMasks, loc, dstBitsType, dstBitsType);
            Value dstBits =
                b.create<hivmave::VFOrOp>(loc, dstBitsType, dstEvenBits, dstOddBits, dstMask, Value()).getResult();
            return b.create<mlir::vector::BitCastOp>(loc, dstVecType, dstBits).getResult();
        }
    }

    // The native fp8<->f32 instructions select one of four byte positions in
    // each 32-bit slot. Keep every RegSlot value intact as pp0..pp3.
    if (isAveFloat8Type(s) && d.isF32())
        return createFp8ToF32Result(b, loc, dstVecType, src, mask, pp);
    if (s.isF32() && isAveFloat8Type(d))
        return createF32ToFp8Result(b, loc, dstVecType, src, mask, rnd, sat, pp);
    // For same-width float<->int conversions the packed even/odd part does not
    // apply (src and dst occupy the full register); pass a null part attribute,
    // matching the arith->AVE lowering.
    hivmave::VCVT_PartTypeAttr partOrNull = (sb == db) ? hivmave::VCVT_PartTypeAttr() : part;

    if (sFloat && dFloat) {
        if (db < sb)
            return b.create<hivmave::VFTruncFOp>(loc, dstVecType, src, mask, rnd, sat, part).getResult();
        // Widening float cast (e.g. f16 -> f32) takes no rounding/saturation.
        return b.create<hivmave::VFExtFOp>(loc, dstVecType, src, mask, part).getResult();
    }
    if (sFloat && !dFloat)
        return b.create<hivmave::VFFpToSIntOp>(loc, dstVecType, src, mask, rnd, sat, partOrNull).getResult();
    if (!sFloat && dFloat) {
        // int -> float: the ISA does not allow #rnd and #part together. A same-width
        // source carries the round mode (rounding may be needed, e.g. i32->f32); a
        // width-changing widen/narrow carries the even/odd part with no round mode
        // (i16->f32 is exact). i64 sources carry both, matching the arith lowering.
        if (sb == db)
            return b.create<hivmave::VFSIntToFpOp>(loc, dstVecType, src, mask, rnd, hivmave::VCVT_PartTypeAttr())
                .getResult();
        if (sb == 64)
            return b.create<hivmave::VFSIntToFpOp>(loc, dstVecType, src, mask, rnd, part).getResult();
        return b.create<hivmave::VFSIntToFpOp>(loc, dstVecType, src, mask, hivm::RoundModeAttr(), part).getResult();
    }
    // int -> int (signed). A 2x width step (e.g. i32<->i16, i16<->i8) uses the
    // even/odd `part`; a 4x step (i32<->i8) uses the pack-pattern `pp` (PP0)
    // instead, matching the arith->AVE lowering. Integer casts do not round.
    auto uni = hivm::UnsignedModeAttr::get(b.getContext(), hivm::UnsignedMode::SI2SI);
    if (db < sb) {
        if (sb / db >= 4)
            return b
                .create<hivmave::VFTruncIOp>(
                    loc, dstVecType, src, mask, sat, hivmave::VCVT_PartTypeAttr(), pp, hivm::UnsignedModeAttr())
                .getResult();
        return b.create<hivmave::VFTruncIOp>(loc, dstVecType, src, mask, sat, part, hivmave::VCVT_PPTypeAttr(), uni)
            .getResult();
    }
    if (db / sb >= 4)
        return b.create<hivmave::VFExtSIOp>(loc, dstVecType, src, mask, hivmave::VCVT_PartTypeAttr(), pp).getResult();
    return b.create<hivmave::VFExtSIOp>(loc, dstVecType, src, mask, part, hivmave::VCVT_PPTypeAttr()).getResult();
}

static FailureOr<Value> createVectorReductionResult(
    OpBuilder& b, Location loc, ::tla::ReduceOp reduceOp, Type elementType, VectorType vecType, Value operand,
    Value explicitMask)
{
    if (failed(validateVectorReduction(reduceOp, elementType)))
        return failure();
    auto aveKind = getAveReductionCombiningKind(reduceOp, elementType);
    if (failed(aveKind))
        return failure();
    // The active mask is supplied explicitly by the frontend; tla.reduce no longer
    // derives one from the operand's originShape.
    if (!explicitMask)
        return reduceOp.emitError("tla.reduce requires an explicit mask"), failure();

    return b.create<hivmave::ReductionOp>(loc, vecType, *aveKind, operand, explicitMask).getResult();
}

static Value createVectorUnaryResult(
    OpBuilder& b, Location loc, VectorUnaryKind kind, Type tlaOperandType, VectorType vecType, Value operand,
    Value mask)
{
    switch (kind) {
        case VectorUnaryKind::Exp:
            return b.create<hivmave::VFExpOp>(loc, vecType, operand, mask, Value()).getResult();
        case VectorUnaryKind::Log:
            return b.create<hivmave::VFLnOp>(loc, vecType, operand, mask, Value()).getResult();
        case VectorUnaryKind::Sqrt:
            return b.create<hivmave::VFSqrtOp>(loc, vecType, operand, mask, Value()).getResult();
        case VectorUnaryKind::Abs:
            return b.create<hivmave::VFAbsOp>(loc, vecType, operand, mask, Value()).getResult();
        case VectorUnaryKind::Neg:
            return b.create<hivmave::VFNegOp>(loc, vecType, operand, mask, Value()).getResult();
        case VectorUnaryKind::Not:
            if (isa<::tla::VectorSSAType>(tlaOperandType))
                return b.create<hivmave::VFNotOp>(loc, vecType, operand, mask, Value()).getResult();
            if (isa<::tla::MaskSSAType>(tlaOperandType))
                return b
                    .create<hivmave::PregNotOp>(
                        loc, vecType, maskWidthAttrForMaskType(b, cast<::tla::MaskSSAType>(tlaOperandType)), operand,
                        mask)
                    .getRes();
            return nullptr;
    }
    return nullptr;
}

// The per-op vector width bundle (one 256-byte register's worth of a given
// element type). Derived fresh for each op from its own operands/result rather
// than shared across the region, so a single vec.func body can mix element
// widths (as tla.cast requires).
struct VecLowerCtx {
    int64_t lanes;
    Type elementType;
    VectorType vecType;
    VectorType maskVecType;
};

// Build the per-op {lanes, elementType, vecType, maskVecType} for a given
// element type. Each op derives its own types this way rather than reusing a
// region-global width: a tla.cast may have produced operands whose element
// width (hence lane count, at a fixed 256-byte register) differs from the
// region's, and same-256-byte register can hold f32 (64), f16 (128) or i8
// (256) lanes.
static FailureOr<VecLowerCtx> deriveVecCtxForElement(Type elementType)
{
    auto lanesOr = getVectorLaneCount(elementType);
    if (failed(lanesOr) || *lanesOr <= 0)
        return failure();
    int64_t lanes = *lanesOr;
    return VecLowerCtx{
        lanes, elementType, VectorType::get({lanes}, elementType), fullPregVecType(elementType.getContext())};
}

// Return the value already mapped into the helper, or clone an arith.constant
// on demand (loop bounds / index math constants are pulled in lazily this way).
static Value lookupOrCloneScalarValue(OpBuilder& b, Value value, DenseMap<Value, Value>& valueMap)
{
    if (Value mapped = valueMap.lookup(value))
        return mapped;
    Operation* def = value.getDefiningOp();
    if (!def || def->getNumResults() != 1)
        return nullptr;
    if (isa<arith::ConstantOp>(def)) {
        Operation* cloned = b.clone(*def);
        valueMap[value] = cloned->getResult(0);
        return cloned->getResult(0);
    }
    return nullptr;
}

// Materialize the descriptor-defined valid-lane count for an omitted store.
// `origin_shape` describes the logical tile extent even when the addressable UB
// subview is a full physical register wide.
static FailureOr<Value> getTlaTensorValidLaneCount(
    OpBuilder& b, Location loc, Value tensorValue, DenseMap<Value, Value>& valueMap)
{
    if (auto descOp = findTensorDescProducer(tensorValue)) {
        Value origin0 = lookupOrCloneScalarValue(b, descOp.getOriginShape0(), valueMap);
        Value origin1 = lookupOrCloneScalarValue(b, descOp.getOriginShape1(), valueMap);
        if (!origin0 || !origin1)
            return failure();
        return b.create<arith::MulIOp>(loc, origin0, origin1).getResult();
    }
    return failure();
}

static std::optional<int64_t> getStaticTlaTensorValidLaneCount(Value tensorValue)
{
    auto descOp = findTensorDescProducer(tensorValue);
    if (!descOp)
        return std::nullopt;
    auto origin0 = getConstantIntValue(descOp.getOriginShape0());
    auto origin1 = getConstantIntValue(descOp.getOriginShape1());
    if (!origin0 || !origin1 || *origin0 < 0 || *origin1 < 0 ||
        (*origin1 != 0 && *origin0 > std::numeric_limits<int64_t>::max() / *origin1))
        return std::nullopt;
    return *origin0 * *origin1;
}

struct IndexInterval {
    int64_t lower;
    int64_t upper;
};

// arith integer operations wrap at their SSA result width. An interval that
// exceeds that signed range cannot be propagated as ordinary int64 arithmetic:
// the wrapped value observed by a later index_cast may be completely different.
static bool intervalFitsValueType(Value value, const IndexInterval& interval)
{
    auto integerType = dyn_cast<IntegerType>(value.getType());
    if (!integerType)
        return true;
    unsigned width = integerType.getWidth();
    if (width == 0 || width > 64)
        return false;
    if (width == 64)
        return true;
    int64_t min = -(int64_t{1} << (width - 1));
    int64_t max = (int64_t{1} << (width - 1)) - 1;
    return interval.lower >= min && interval.upper <= max;
}

// Evaluate the small affine subset produced by tiled vector loops. This is
// deliberately conservative: unknown values retain the PLT tail path.
static std::optional<IndexInterval> getIndexInterval(Value value)
{
    if (std::optional<int64_t> constant = getConstantIntValue(value))
        return IndexInterval{*constant, *constant};

    if (auto arg = dyn_cast<BlockArgument>(value)) {
        auto forOp = dyn_cast_or_null<scf::ForOp>(arg.getOwner()->getParentOp());
        if (!forOp || arg.getArgNumber() != 0)
            return std::nullopt;
        auto lowerInterval = getIndexInterval(forOp.getLowerBound());
        auto upperInterval = getIndexInterval(forOp.getUpperBound());
        auto stepInterval = getIndexInterval(forOp.getStep());
        if (!lowerInterval || lowerInterval->lower != lowerInterval->upper || !upperInterval ||
            upperInterval->lower != upperInterval->upper || !stepInterval || stepInterval->lower != stepInterval->upper)
            return std::nullopt;
        int64_t lower = lowerInterval->lower;
        int64_t upper = upperInterval->lower;
        int64_t step = stepInterval->lower;
        if (step <= 0 || upper <= lower)
            return std::nullopt;
        int64_t span;
        int64_t lastOffset;
        int64_t last;
        if (__builtin_sub_overflow(upper, lower, &span) || __builtin_sub_overflow(span, int64_t{1}, &span) ||
            __builtin_mul_overflow(span / step, step, &lastOffset) || __builtin_add_overflow(lower, lastOffset, &last))
            return std::nullopt;
        IndexInterval result{lower, last};
        if (intervalFitsValueType(value, result))
            return result;
        return std::nullopt;
    }
    if (auto cast = value.getDefiningOp<arith::IndexCastOp>()) {
        auto interval = getIndexInterval(cast.getIn());
        if (!interval)
            return std::nullopt;
        if (auto destType = dyn_cast<IntegerType>(cast.getType())) {
            unsigned width = destType.getWidth();
            if (width == 0 || width > 64)
                return std::nullopt;
            int64_t min = width == 64 ? std::numeric_limits<int64_t>::min() : -(int64_t{1} << (width - 1));
            int64_t max = width == 64 ? std::numeric_limits<int64_t>::max() : (int64_t{1} << (width - 1)) - 1;
            if (interval->lower < min || interval->upper > max)
                return std::nullopt;
        }
        return interval;
    }
    if (auto add = value.getDefiningOp<arith::AddIOp>()) {
        auto lhs = getIndexInterval(add.getLhs());
        auto rhs = getIndexInterval(add.getRhs());
        int64_t lower;
        int64_t upper;
        if (lhs && rhs && !__builtin_add_overflow(lhs->lower, rhs->lower, &lower) &&
            !__builtin_add_overflow(lhs->upper, rhs->upper, &upper)) {
            IndexInterval result{lower, upper};
            if (intervalFitsValueType(value, result))
                return result;
        }
    }
    if (auto sub = value.getDefiningOp<arith::SubIOp>()) {
        auto lhs = getIndexInterval(sub.getLhs());
        auto rhs = getIndexInterval(sub.getRhs());
        int64_t lower;
        int64_t upper;
        if (lhs && rhs && !__builtin_sub_overflow(lhs->lower, rhs->upper, &lower) &&
            !__builtin_sub_overflow(lhs->upper, rhs->lower, &upper)) {
            IndexInterval result{lower, upper};
            if (intervalFitsValueType(value, result))
                return result;
        }
    }
    if (auto mul = value.getDefiningOp<arith::MulIOp>()) {
        auto lhs = getIndexInterval(mul.getLhs());
        auto rhs = getIndexInterval(mul.getRhs());
        int64_t lower;
        int64_t upper;
        if (lhs && rhs && lhs->lower >= 0 && rhs->lower >= 0 &&
            !__builtin_mul_overflow(lhs->lower, rhs->lower, &lower) &&
            !__builtin_mul_overflow(lhs->upper, rhs->upper, &upper)) {
            IndexInterval result{lower, upper};
            if (intervalFitsValueType(value, result))
                return result;
        }
    }
    if (auto min = value.getDefiningOp<arith::MinSIOp>()) {
        auto lhs = getIndexInterval(min.getLhs());
        auto rhs = getIndexInterval(min.getRhs());
        if (lhs && rhs) {
            IndexInterval result{std::min(lhs->lower, rhs->lower), std::min(lhs->upper, rhs->upper)};
            if (intervalFitsValueType(value, result))
                return result;
        }
    }
    return std::nullopt;
}

static bool isProvablyFullVectorStore(::tla::StoreOp storeOp, int64_t lanes)
{
    auto descOp = findTensorDescProducer(storeOp.getDest());
    if (!descOp)
        return false;
    auto origin0 = getIndexInterval(descOp.getOriginShape0());
    auto origin1 = getIndexInterval(descOp.getOriginShape1());
    if (!origin0 || !origin1 || origin0->lower <= 0 || origin1->lower <= 0)
        return false;
    int64_t requiredRows = lanes / origin1->lower;
    if (lanes % origin1->lower != 0)
        ++requiredRows;
    return origin0->lower >= requiredRows;
}

// Packed stores use the source-predicate-to-destination-element geometry
// validated by tla.store. Keep this defensive lookup in the lowering too: a
// malformed program may reach the pass through a nonstandard pipeline.
static FailureOr<int64_t> getStorePredicateDestLaneMultiplier(::tla::StoreOp storeOp, Type sourceElementType)
{
    auto storeDist = storeOp.getStoreDist();
    ::StoreDist dist = storeDist ? storeDist->getStoreDist() : ::StoreDist::norm;
    auto destType = dyn_cast<::tla::TlaTensorType>(storeOp.getDest().getType());
    if (!destType)
        return failure();
    auto geometry = getPackedStorePredicateGeometry(dist, sourceElementType, destType.getPtr().getPointee());
    if (!geometry)
        return failure();
    return geometry->sourcePredicateLanesPerDestElement;
}

// An omitted vector-store mask is defined by the destination tile alone. Use
// the shared ALL predicate for a destination proven to cover the physical
// store width; retain a PLT tail predicate otherwise. Source-producing ops do
// not affect this default: callers use an explicit store mask for sparse or
// otherwise non-default writes.
static FailureOr<Value> createImplicitVectorStoreMask(
    OpBuilder& b, Location loc, ::tla::StoreOp storeOp, VecLowerCtx& opCtx, DenseMap<Value, Value>& valueMap,
    AutoFullPredicateCache& fullMasks)
{
    auto storeDist = storeOp.getStoreDist();
    ::StoreDist dist = storeDist ? storeDist->getStoreDist() : ::StoreDist::norm;
    auto destLaneMultiplierOr = getStorePredicateDestLaneMultiplier(storeOp, opCtx.elementType);
    if (failed(destLaneMultiplierOr))
        return storeOp.emitError("unsupported packed-store predicate geometry"), failure();
    int64_t destLaneMultiplier = *destLaneMultiplierOr;

    // The supported vector frontend has no 64-bit on-chip element type. Raw
    // i64 IR historically used a B32 PLT predicate for omitted masks; keep that
    // behavior instead of introducing an unproven B32 PGE encoding for B64
    // stores. Proper B64 predicate support is separate work.
    bool supportsImplicitAll = opCtx.elementType.getIntOrFloatBitWidth() <= 32;

    // FIRST_ELEMENT modes ignore predicate bits in the backend. Supplying a
    // destination-tail PLT would only add work and suggest semantics that the
    // instruction does not implement, so omitted masks use the shared ALL
    // singleton directly.
    if (supportsImplicitAll && (dist == ::StoreDist::first_element_b8 || dist == ::StoreDist::first_element_b16 ||
                                dist == ::StoreDist::first_element_b32))
        return fullMasks.get(loc, opCtx.lanes);

    int64_t requiredDestLanes = (opCtx.lanes + destLaneMultiplier - 1) / destLaneMultiplier;
    auto destStaticLanes = getStaticTlaTensorValidLaneCount(storeOp.getDest());

    if (supportsImplicitAll && isProvablyFullVectorStore(storeOp, requiredDestLanes))
        return fullMasks.get(loc, opCtx.lanes);

    if (destStaticLanes) {
        int64_t maxDestLanes = opCtx.lanes / destLaneMultiplier;
        int64_t activeLanes = *destStaticLanes >= maxDestLanes ? opCtx.lanes : *destStaticLanes * destLaneMultiplier;
        Value active = b.create<arith::ConstantIndexOp>(loc, activeLanes);
        return createPredicatePlt(b, loc, opCtx.maskVecType, opCtx.lanes, active).getRes();
    }

    auto destLanes = getTlaTensorValidLaneCount(b, loc, storeOp.getDest(), valueMap);
    if (failed(destLanes))
        return failure();
    Value destLimit = *destLanes;
    if (destLaneMultiplier != 1) {
        Value multiplier = b.create<arith::ConstantIndexOp>(loc, destLaneMultiplier);
        destLimit = b.create<arith::MulIOp>(loc, destLimit, multiplier);
    }
    return createPredicatePlt(b, loc, opCtx.maskVecType, opCtx.lanes, destLimit).getRes();
}

static FailureOr<Value> castScalarForVectorElement(Value scalar, Type elementType)
{
    if (scalar.getType() == elementType)
        return scalar;
    return failure();
}

static FailureOr<Value> materializeVectorScalarValue(
    OpBuilder& b, VectorOpInfo info, DenseMap<Value, Value>& valueMap, VecLowerCtx& ctx)
{
    TlaBinaryOperands operands = info.operands;
    Value scalar = lookupOrCloneScalarValue(b, operands.rhs, valueMap);
    if (!scalar)
        return failure();
    // Shift scalar ops deliberately carry i16, matching the vshls/vshrs
    // intrinsic ABI rather than the source vector element type.
    if (info.kind == VectorBinaryKind::ShiftLeft || info.kind == VectorBinaryKind::ShiftRight)
        return scalar;
    auto castScalar = castScalarForVectorElement(scalar, ctx.elementType);
    if (failed(castScalar))
        return failure();
    return *castScalar;
}

static FailureOr<Value> createVectorScalarBinaryResult(
    OpBuilder& b, Location loc, VectorOpInfo info, VecLowerCtx& ctx, Value lhs, Value scalar, Value mask)
{
    if (info.kind == VectorBinaryKind::ShiftLeft || info.kind == VectorBinaryKind::ShiftRight) {
        if (!isa<IntegerType>(ctx.elementType) || !scalar.getType().isInteger(16))
            return failure();
        if (info.kind == VectorBinaryKind::ShiftLeft)
            return b.create<hivm_regbaseintrins::VshlsSXInstrOp>(loc, ctx.vecType, lhs, scalar, mask).getRes();
        return b.create<hivm_regbaseintrins::VshrsSXInstrOp>(loc, ctx.vecType, lhs, scalar, mask).getRes();
    }

    if (info.kind == VectorBinaryKind::Add || info.kind == VectorBinaryKind::Mul ||
        info.kind == VectorBinaryKind::Max || info.kind == VectorBinaryKind::Min) {
        if (info.kind == VectorBinaryKind::Add)
            return b.create<hivmave::VFAddsOp>(loc, ctx.vecType, lhs, scalar, mask, Value()).getResult();
        if (info.kind == VectorBinaryKind::Mul)
            return b.create<hivmave::VFMulsOp>(loc, ctx.vecType, lhs, scalar, mask, Value()).getResult();
        if (info.kind == VectorBinaryKind::Max)
            return b.create<hivmave::VFMaxsOp>(loc, ctx.vecType, lhs, scalar, mask, Value()).getResult();
        return b.create<hivmave::VFMinsOp>(loc, ctx.vecType, lhs, scalar, mask, Value()).getResult();
    }

    Value rhs = b.create<hivmave::VFBroadcastScalarOp>(loc, ctx.vecType, scalar).getRes();
    return createVectorBinaryResult(
        b, loc, info.kind, info.operands.lhs.getType(), ctx.elementType, ctx.vecType, lhs, rhs, mask);
}

static FailureOr<Type> lowerSCFCarrierType(Type type)
{
    if (auto vectorType = dyn_cast<::tla::VectorSSAType>(type)) {
        auto ctx = deriveVecCtxForElement(vectorType.getElementType());
        if (failed(ctx))
            return failure();
        return Type(ctx->vecType);
    }
    if (isa<::tla::MaskSSAType>(type))
        return Type(fullPregVecType(type.getContext()));
    return type;
}

static LogicalResult lowerNestedVectorBlock(
    Block* sourceBlock, OpBuilder& b, ModuleOp module, DenseMap<Value, Value>& valueMap,
    AutoFullPredicateCache& fullMasks);

// Materialize one tla.tensor_desc as an addressable subview inside the helper.
//
// tla-lower-tensor-desc is the sole descriptor producer, so the descriptor is
// consumed directly: its coord / stride slots carry the position and pitch and
// there is no producer chain to re-walk. Carves a lanes-wide (256-byte) flat
// subview of the helper's base-memref argument at the flat offset
// coord0 * stride0 + coord1 * stride1.
static LogicalResult materializeTensorDescSubview(
    ::tla::TensorDescOp descOp, OpBuilder& b, DenseMap<Value, Value>& valueMap)
{
    Location loc = descOp.getLoc();
    Value baseMemref = valueMap.lookup(descOp.getResult());
    if (!baseMemref)
        return descOp.emitError("failed to map tla.tensor_desc base in vector helper"), failure();
    auto sourceType = dyn_cast<MemRefType>(baseMemref.getType());
    if (!sourceType || sourceType.getRank() != 1)
        return descOp.emitError("expected rank-1 base memref for vector tensor_desc"), failure();
    auto lanesOr = getVectorLaneCount(sourceType.getElementType());
    if (failed(lanesOr))
        return descOp.emitError("unsupported element type for vector tensor_desc"), failure();
    Value rowOff = lookupOrCloneScalarValue(b, descOp.getCoord0(), valueMap);
    Value colOff = lookupOrCloneScalarValue(b, descOp.getCoord1(), valueMap);
    Value stride0 = lookupOrCloneScalarValue(b, descOp.getStride0(), valueMap);
    Value stride1 = lookupOrCloneScalarValue(b, descOp.getStride1(), valueMap);
    auto info = decodeTensorTypeInfo(descOp.getResult().getType());
    if (failed(info))
        return failure();
    if (!rowOff || !colOff || !stride0 || !stride1)
        return descOp.emitError(
                   "tensor_desc offset or stride is not reachable "
                   "from the vector helper: it is neither a constant "
                   "nor a captured scalar argument"),
               failure();
    Value flatOffset;
    if (isLinearLayout(info->layoutTag)) {
        flatOffset = b.create<arith::AddIOp>(
            loc, b.create<arith::MulIOp>(loc, rowOff, stride0), b.create<arith::MulIOp>(loc, colOff, stride1));
    } else if (isNZFamilyLayout(info->layoutTag)) {
        auto shape0 = lookupOrCloneScalarValue(b, descOp.getShape0(), valueMap);
        auto shape1 = lookupOrCloneScalarValue(b, descOp.getShape1(), valueMap);
        auto shape2 = lookupOrCloneScalarValue(b, descOp.getShape2(), valueMap);
        auto shape3 = lookupOrCloneScalarValue(b, descOp.getShape3(), valueMap);
        stride0 = lookupOrCloneScalarValue(b, descOp.getStride0(), valueMap);
        stride1 = lookupOrCloneScalarValue(b, descOp.getStride1(), valueMap);
        auto stride2 = lookupOrCloneScalarValue(b, descOp.getStride2(), valueMap);
        auto stride3 = lookupOrCloneScalarValue(b, descOp.getStride3(), valueMap);
        if (!shape0 || !shape1 || !shape2 || !shape3 || !stride0 || !stride1 || !stride2 || !stride3)
            return failure();
        if (info->layoutTag == LayoutTag::zN) {
            Value pc0 = b.create<arith::RemSIOp>(loc, rowOff, shape0);
            Value pc1 = b.create<arith::DivSIOp>(loc, rowOff, shape0);
            Value pc2 = b.create<arith::RemSIOp>(loc, colOff, shape2);
            Value pc3 = b.create<arith::DivSIOp>(loc, colOff, shape2);
            Value t0 = b.create<arith::MulIOp>(loc, pc0, stride0);
            Value t1 = b.create<arith::MulIOp>(loc, pc1, stride1);
            Value t2 = b.create<arith::MulIOp>(loc, pc2, stride2);
            Value t3 = b.create<arith::MulIOp>(loc, pc3, stride3);
            Value sum01 = b.create<arith::AddIOp>(loc, t0, t1);
            Value sum23 = b.create<arith::AddIOp>(loc, t2, t3);
            flatOffset = b.create<arith::AddIOp>(loc, sum01, sum23);
        } else if (info->layoutTag == LayoutTag::zNUnAlign) {
            // zNUnAlign only used in on-chip memory, and the scalar unit in a
            // vf only supports integer arith to 32 bits. Keep the entire
            // address calculation in i32, then convert its final result to the
            // index type required by memref.reinterpret_cast.
            Type i32Type = b.getI32Type();
            Value rowOffI32 = b.create<arith::IndexCastOp>(loc, i32Type, rowOff);
            Value colOffI32 = b.create<arith::IndexCastOp>(loc, i32Type, colOff);
            Value shape2I32 = b.create<arith::IndexCastOp>(loc, i32Type, shape2);
            Value stride0I32 = b.create<arith::IndexCastOp>(loc, i32Type, stride0);
            Value stride1I32 = b.create<arith::IndexCastOp>(loc, i32Type, stride1);
            Value stride2I32 = b.create<arith::IndexCastOp>(loc, i32Type, stride2);
            Value stride3I32 = b.create<arith::IndexCastOp>(loc, i32Type, stride3);
            Value shape0OrgI32 = b.create<arith::DivSIOp>(loc, stride3I32, shape2I32);
            Value pc0I32 = b.create<arith::RemSIOp>(loc, rowOffI32, shape0OrgI32);
            Value pc1I32 = b.create<arith::DivSIOp>(loc, rowOffI32, shape0OrgI32);
            Value pc2I32 = b.create<arith::RemSIOp>(loc, colOffI32, shape2I32);
            Value pc3I32 = b.create<arith::DivSIOp>(loc, colOffI32, shape2I32);
            Value t0 = b.create<arith::MulIOp>(loc, pc0I32, stride0I32);
            Value t1 = b.create<arith::MulIOp>(loc, pc1I32, stride1I32);
            Value t2 = b.create<arith::MulIOp>(loc, pc2I32, stride2I32);
            Value t3 = b.create<arith::MulIOp>(loc, pc3I32, stride3I32);
            Value sum01 = b.create<arith::AddIOp>(loc, t0, t1);
            Value sum23 = b.create<arith::AddIOp>(loc, t2, t3);
            Value flatOffsetI32 = b.create<arith::AddIOp>(loc, sum01, sum23);
            flatOffset = b.create<arith::IndexCastOp>(loc, b.getIndexType(), flatOffsetI32);
        } else {
            return descOp->emitError() << "unsupported NZFamily layout to get flatOffset";
        }
    } else {
        return descOp->emitError() << "unsupported layout to get flatOffset";
    }
    valueMap[descOp.getResult()] = ::tla::materializeFlatReinterpretSubview(b, loc, baseMemref, flatOffset, *lanesOr);
    return success();
}

// Re-create one vec.func body op inside the helper: tla ops become AVE vector
// ops; scf control flow and index arithmetic are carried verbatim. Each op
// derives its own vector/mask width from its operands or result element type,
// so a single region may mix element widths (e.g. across tla.cast).
static LogicalResult lowerNestedVectorOp(
    Operation& op, OpBuilder& b, ModuleOp module, DenseMap<Value, Value>& valueMap, AutoFullPredicateCache& fullMasks)
{
    Location loc = op.getLoc();

    // make_shape / make_coord are dead after tla-lower-tensor-desc (their leaves
    // were folded into tensor_desc operands); skip them. (tla-finalize-memref
    // erases them.)
    if (isa<::tla::MakeShapeOp, ::tla::MakeCoordOp>(op))
        return success();

    if (auto constant = dyn_cast<arith::ConstantOp>(op)) {
        valueMap[constant.getResult()] = b.clone(op)->getResult(0);
        return success();
    }

    // tla.tensor_desc: materialize it as a subview of the helper's base memref
    // argument (see materializeTensorDescSubview).
    if (auto descOp = dyn_cast<::tla::TensorDescOp>(op))
        return materializeTensorDescSubview(descOp, b, valueMap);

    if (auto loadOp = dyn_cast<::tla::LoadOp>(op)) {
        Value source = valueMap.lookup(loadOp.getSource());
        if (!source)
            return failure();

        // MaskSSA result: 1/2/4-byte UB → i1 memref view → vload <NORM> (plds.b8).
        if (auto maskType = dyn_cast<::tla::MaskSSAType>(loadOp.getResult().getType())) {
            int64_t lanes = maskType.getPhysicalLanes();
            auto i1MemrefOr = materializeI1MaskMemrefFromUb(b, loc, source, lanes);
            if (failed(i1MemrefOr))
                return loadOp.emitError("failed to materialize i1 memref view for tla.load MaskSSA"), failure();
            VectorType semanticMaskType = VectorType::get({lanes}, b.getI1Type());
            Value zero = b.create<arith::ConstantIndexOp>(loc, 0);
            auto vfLoad =
                createVFLoad(b, loc, semanticMaskType, *i1MemrefOr, zero, hivmave::LoadDist::NORM, /*unaligned=*/false);
            Value loaded = castMaskToPregType(b, loc, vfLoad.getRes(), fullPregVecType(b.getContext()));
            valueMap[loadOp.getResult()] = loaded;
            return success();
        }

        // The loaded vector's element type comes from the tile memref, not the
        // region-global width: a load feeding a differently-typed op keeps its own
        // dtype.
        auto sourceType = dyn_cast<MemRefType>(source.getType());
        if (!sourceType)
            return failure();
        auto opCtx = deriveVecCtxForElement(sourceType.getElementType());
        if (failed(opCtx))
            return failure();
        Value zero = b.create<arith::ConstantIndexOp>(loc, 0);
        hivmave::LoadDist pattern = hivmave::LoadDist::NORM;
        if (auto loadDistAttr = loadOp.getLoadDist())
            pattern = mapTlaLoadDistToAve(loadDistAttr->getLoadDist());
        bool dual = isDualDestLoadDist(pattern);
        if (dual != static_cast<bool>(loadOp.getResult2()))
            return loadOp.emitError(
                       "dintlv load_dist requires exactly two results; other "
                       "load_dist values require one"),
                   failure();
        // Strided block load (vsldb): lower through a library call like the
        // store-with-stride path. The tile memref supplies the base address;
        // the gather reads all 8 DataBlocks, so the predicate is all-true.
        if (auto blockStrideAttr = loadOp.getBlockStrideAttr()) {
            Value blockStrideVal = b.create<arith::ConstantIntOp>(loc, blockStrideAttr.getInt(), 32);
            int64_t repeatStrideImm = loadOp.getRepeatStrideAttr() ? loadOp.getRepeatStrideAttr().getInt() : 0;
            Value repeatStrideVal = b.create<arith::ConstantIntOp>(loc, repeatStrideImm, 32);
            std::string calleeName = getLoadWithStrideLibraryCallName(sourceType.getElementType());
            if (calleeName.empty())
                return loadOp.emitError("unsupported element type for tla.load with block_stride: ")
                           << sourceType.getElementType(),
                       failure();
            bool viaBytes = blockStrideUsesByteRegister(sourceType.getElementType());
            Type resultVecType = viaBytes ? byteRegisterTypeFor(opCtx->vecType) : Type(opCtx->vecType);
            auto callee =
                getOrCreateLoadWithStrideLibraryCall(module, loc, resultVecType, source.getType(), calleeName);
            if (!callee)
                return failure();
            // The gather reads all 8 DataBlocks, so the predicate is all-true;
            // build it via PgePattern::ALL like the other no-mask lowerings
            // (a dense<true> arith constant is not selectable in bisheng).
            Value allTrue = allTrueMaskFor(fullMasks, loc, opCtx->vecType, loadOp.getResult().getType());
            auto call =
                b.create<func::CallOp>(loc, callee, ValueRange{source, blockStrideVal, repeatStrideVal, allTrue});
            Value loaded = call.getResult(0);
            if (viaBytes)
                loaded = b.create<mlir::vector::BitCastOp>(loc, opCtx->vecType, loaded).getResult();
            valueMap[loadOp.getResult()] = loaded;
            return success();
        }
        // DINTLV_* still takes a VL-wide tile view (same as AVE ProcessVsstb /
        // HIVM2VLLoadOpLowering): the distribution pattern reads 2*VL from that
        // base address. Do not widen the memref to 2*VL — that breaks hivmc ABI.
        auto vfLoad =
            createVFLoad(b, loc, opCtx->vecType, source, zero, pattern, loadOp.getUnalignedUbAccess().value_or(false));
        valueMap[loadOp.getResult()] = vfLoad.getRes();
        if (dual)
            valueMap[loadOp.getResult2()] = vfLoad.getRes1();
        return success();
    }

    // Preserve local-memory ordering while outlining a vec.func into its AVE
    // helper. Every operation in the source region must be recreated explicitly;
    // otherwise helper construction fails and leaves the whole vec.func behind.
    if (auto localMemBarOp = dyn_cast<::tla::LocalMemBarOp>(op)) {
        return lowerLocalMemBar(b, localMemBarOp);
    }

    // tla.cast: element-type conversion. The source vector already carries its
    // width; the destination width is one full 256-byte register's worth of the
    // target element type. The cast op picks the AVE cast (vtruncf / vfptosi /
    // vsitofp / vtrunci / ...) from the (src,dst) element kinds.

    if (auto castOp = dyn_cast<::tla::CastOp>(op)) {
        Value src = valueMap.lookup(castOp.getSource());
        if (!src)
            return failure();
        auto srcVecType = dyn_cast<VectorType>(src.getType());
        if (!srcVecType)
            return castOp.emitError("tla.cast source is not a vector value"), failure();
        auto dstType = dyn_cast<::tla::VectorSSAType>(castOp.getResult().getType());
        if (!dstType)
            return castOp.emitError("expected !tla.vector result type"), failure();
        auto dstLanesOr = getVectorLaneCount(dstType.getElementType());
        if (failed(dstLanesOr))
            return castOp.emitError("unsupported tla.cast destination element type"), failure();
        auto dstVecType = VectorType::get({*dstLanesOr}, dstType.getElementType());
        // Reject casts whose source or destination element type has no AVE cast path
        // (unsigned integers, i1/bool, f64) rather than emitting invalid AVE IR.
        if (!isSupportedCastElementType(srcVecType.getElementType()) ||
            !isSupportedCastElementType(dstVecType.getElementType()))
            return castOp.emitError(
                       "unsupported tla.cast element type: only signed "
                       "integers (i8/i16/i32/i64) and floats (f16/bf16/f32) "
                       "are supported; unsigned, bool and f64 are not"),
                   failure();
        ArrayRef<int32_t> trait = castOp.getTrait();
        if (trait.size() != 3)
            return castOp.emitError("tla.cast trait must have 3 codes"), failure();
        // An optional mask predicates the source lanes of the AVE cast; all-true when
        // none is given.
        Value mask;
        if (castOp.getMask()) {
            mask = valueMap.lookup(castOp.getMask());
            if (!mask)
                return failure();
        } else {
            mask = allTrueMaskFor(fullMasks, loc, srcVecType, castOp.getSource().getType());
        }
        auto result =
            createVectorCastResult(b, fullMasks, loc, srcVecType, dstVecType, trait, src, mask, (bool)castOp.getMask());
        if (failed(result))
            return castOp.emitError("unsupported tla.cast element type conversion"), failure();
        valueMap[castOp.getResult()] = *result;
        return success();
    }

    if (auto fullOp = dyn_cast<::tla::FullOp>(op)) {
        // No vector operand to key off: the broadcast width comes from the result
        // VectorSSA element type.
        auto resultType = dyn_cast<::tla::VectorSSAType>(fullOp.getResult().getType());
        if (!resultType)
            return fullOp.emitError("expected !tla.vector result type"), failure();
        auto opCtx = deriveVecCtxForElement(resultType.getElementType());
        if (failed(opCtx))
            return fullOp.emitError("unsupported tla.full result element type"), failure();
        // An optional tla mask predicates the broadcast lanes; all-true when absent.
        Value mask;
        if (fullOp.getMask()) {
            mask = valueMap.lookup(fullOp.getMask());
            if (!mask)
                return failure();
        } else {
            mask = allTrueMaskFor(fullMasks, loc, opCtx->vecType, fullOp.getResult().getType());
        }

        // A one-lane vector fragment source (e.g. a tla.reduce result) is already
        // register-resident; broadcast lane 0 across the register with AVE
        // vector_broadcast. This stays entirely in the vector register file and
        // avoids reading a vector lane into a scalar register, which the SIMD
        // backend cannot legalize.
        if (Value mapped = valueMap.lookup(fullOp.getValue())) {
            if (auto srcVecType = dyn_cast<VectorType>(mapped.getType())) {
                if (srcVecType.getElementType() != opCtx->elementType)
                    return fullOp.emitError("tla.full source element type ")
                               << srcVecType.getElementType() << " does not match vector element type "
                               << opCtx->elementType,
                           failure();
                valueMap[fullOp.getResult()] =
                    b.create<hivmave::VFBroadcastVectorOp>(loc, opCtx->vecType, mapped, mask, true).getRes();
                return success();
            }
        }

        // A plain scalar source broadcasts across the register via AVE broadcast
        // (mask-capable scalar splat, lowers to vdups.z). i1 never reaches here:
        // it is absent from the op's accepted value types, and an i1 result type
        // fails deriveVecCtxForElement above.
        Value source = lookupOrCloneScalarValue(b, fullOp.getValue(), valueMap);
        if (!source)
            return failure();
        if (source.getType() != opCtx->elementType)
            return fullOp.emitError("tla.full scalar type ")
                       << source.getType() << " does not match vector element type " << opCtx->elementType,
                   failure();
        valueMap[fullOp.getResult()] =
            b.create<hivmave::VFBroadcastScalarMaskOp>(loc, opCtx->vecType, source, mask).getRes();
        return success();
    }

    if (auto arangeOp = dyn_cast<::tla::ArangeOp>(op)) {
        Value start = lookupOrCloneScalarValue(b, arangeOp.getStart(), valueMap);
        if (!start)
            return failure();
        // Width comes from the result VectorSSA element type.
        auto resultType = dyn_cast<::tla::VectorSSAType>(arangeOp.getResult().getType());
        if (!resultType)
            return arangeOp.emitError("expected !tla.vector result type"), failure();
        auto opCtx = deriveVecCtxForElement(resultType.getElementType());
        if (failed(opCtx))
            return arangeOp.emitError("unsupported tla.arange result element type"), failure();
        if (isa<FloatType>(opCtx->elementType))
            return arangeOp.emitError("tla.arange does not support floating-point element types"), failure();
        if (start.getType() != opCtx->elementType)
            return arangeOp.emitError("tla.arange start type ")
                       << start.getType() << " does not match vector element type " << opCtx->elementType,
                   failure();
        auto vciType = hivmave::VCIType::INCREASE;
        if (arangeOp.getOrder() == "decrease")
            vciType = hivmave::VCIType::DECREASE;
        else if (arangeOp.getOrder() != "increase")
            return arangeOp.emitError("unsupported tla.arange order: ") << arangeOp.getOrder(), failure();
        valueMap[arangeOp.getResult()] =
            b.create<hivmave::VFVCIOp>(loc, opCtx->vecType, start, hivmave::VCITypeAttr::get(b.getContext(), vciType))
                .getRes();
        return success();
    }

    if (auto info = getVectorBinaryInfo(&op)) {
        if (op.getNumResults() != 1)
            return failure();
        TlaBinaryOperands operands = info->operands;
        Value lhs = valueMap.lookup(operands.lhs);
        if (!lhs)
            return failure();
        Value rhs = valueMap.lookup(operands.rhs);
        if (!rhs)
            return failure();
        // Derive the vector width from the operands: a cast may have produced a
        // vector of a different lane width than the enclosing region's element type.
        auto opVecType = dyn_cast<VectorType>(lhs.getType());
        if (!opVecType)
            return failure();
        Type opElemType = opVecType.getElementType();
        Value mask;
        if (operands.mask) {
            mask = valueMap.lookup(operands.mask);
            if (!mask)
                return failure();
        } else {
            mask = allTrueMaskFor(fullMasks, loc, opVecType, operands.lhs.getType());
        }
        Value result =
            createVectorBinaryResult(b, loc, info->kind, operands.lhs.getType(), opElemType, opVecType, lhs, rhs, mask);
        if (!result)
            return failure();
        valueMap[op.getResult(0)] = result;
        return success();
    }

    if (auto info = getVectorScalarBinaryInfo(&op)) {
        if (op.getNumResults() != 1)
            return failure();
        TlaBinaryOperands operands = info->operands;
        Value lhs = valueMap.lookup(operands.lhs);
        if (!lhs)
            return failure();
        // Element type follows the lhs vector operand, not the region width.
        auto lhsTy = dyn_cast<VectorType>(lhs.getType());
        if (!lhsTy)
            return failure();
        auto opCtx = deriveVecCtxForElement(lhsTy.getElementType());
        if (failed(opCtx))
            return failure();
        auto scalarOr = materializeVectorScalarValue(b, *info, valueMap, *opCtx);
        if (failed(scalarOr))
            return failure();
        Value mask;
        if (operands.mask) {
            mask = valueMap.lookup(operands.mask);
            if (!mask)
                return failure();
        } else {
            mask = fullMasks.get(loc, opCtx->lanes);
        }
        auto result = createVectorScalarBinaryResult(b, loc, *info, *opCtx, lhs, *scalarOr, mask);
        if (failed(result))
            return failure();
        valueMap[op.getResult(0)] = *result;
        return success();
    }

    // tla.where: per-lane select. The mask controls which lanes take `x`; the
    // remaining lanes take `y`. Lowers to ave.hir.vsel(mask, x, y).
    if (auto whereOp = dyn_cast<::tla::WhereOp>(op)) {
        Value mask = valueMap.lookup(whereOp.getMask());
        Value x = valueMap.lookup(whereOp.getX());
        Value y = valueMap.lookup(whereOp.getY());
        if (!mask || !x || !y)
            return failure();
        // Result width follows the selected vectors' element type.
        auto xTy = dyn_cast<VectorType>(x.getType());
        if (!xTy)
            return failure();
        auto opCtx = deriveVecCtxForElement(xTy.getElementType());
        if (failed(opCtx))
            return failure();
        valueMap[whereOp.getResult()] = b.create<hivmave::VFSelectOp>(loc, opCtx->vecType, mask, x, y);
        return success();
    }

    // tla.squeeze: mask-compress src lanes via linked bitcode (vsqz). Uses
    // NO_STORE_REG; STORE_REG + StoreUnAlign streaming writeback is not exposed
    // until unaligned store (StoreUnAlign/StoreUnAlignPost) is available in TLA.
    if (auto squeezeOp = dyn_cast<::tla::SqueezeOp>(op)) {
        Value src = valueMap.lookup(squeezeOp.getSrc());
        Value mask = valueMap.lookup(squeezeOp.getMask());
        if (!src || !mask)
            return failure();
        auto srcTy = dyn_cast<VectorType>(src.getType());
        if (!srcTy)
            return failure();
        auto opCtx = deriveVecCtxForElement(srcTy.getElementType());
        if (failed(opCtx))
            return failure();
        std::string calleeName = getSqueezeLibraryCallName(srcTy.getElementType());
        if (calleeName.empty())
            return squeezeOp.emitError("unsupported element type for tla.squeeze: ") << srcTy.getElementType(),
                   failure();
        VectorType pregVecType = fullPregVecType(b.getContext());
        Value preg = castMaskToPregType(b, loc, mask, pregVecType);
        auto callee = getOrCreateSqueezeLibraryCall(module, loc, opCtx->vecType, pregVecType, calleeName);
        Value result = b.create<func::CallOp>(loc, callee, ValueRange{src, preg}).getResult(0);
        valueMap[squeezeOp.getResult()] = result;
        return success();
    }

    if (auto reduceOp = dyn_cast<::tla::ReduceOp>(op)) {
        if (op.getNumResults() != 1)
            return failure();
        Value operand = valueMap.lookup(reduceOp->getOperand(0));
        if (!operand)
            return failure();
        // Reduction width follows the operand vector's element type.
        auto operandTy = dyn_cast<VectorType>(operand.getType());
        if (!operandTy)
            return failure();
        auto opCtx = deriveVecCtxForElement(operandTy.getElementType());
        if (failed(opCtx))
            return failure();
        Value mask;
        if (reduceOp.getMask()) {
            mask = valueMap.lookup(reduceOp.getMask());
            if (!mask)
                return failure();
        }
        auto result = createVectorReductionResult(b, loc, reduceOp, opCtx->elementType, opCtx->vecType, operand, mask);
        if (failed(result))
            return failure();
        valueMap[op.getResult(0)] = *result;
        return success();
    }

    if (auto interleaveOp = dyn_cast<::tla::InterleaveOp>(op)) {
        if (op.getNumResults() != 2)
            return failure();

        Value src0 = valueMap.lookup(interleaveOp.getSrc0());
        Value src1 = valueMap.lookup(interleaveOp.getSrc1());
        if (!src0 || !src1)
            return failure();

        auto src0Type = dyn_cast<VectorType>(src0.getType());
        auto src1Type = dyn_cast<VectorType>(src1.getType());
        if (!src0Type || !src1Type || src0Type != src1Type)
            return failure();

        auto aveOp = b.create<hivmave::VFInterleaveOp>(loc, TypeRange{src0Type, src1Type}, ValueRange{src0, src1});

        valueMap[op.getResult(0)] = aveOp->getResult(0);
        valueMap[op.getResult(1)] = aveOp->getResult(1);
        return success();
    }

    if (auto deInterleaveOp = dyn_cast<::tla::DeInterleaveOp>(op)) {
        if (op.getNumResults() != 2)
            return failure();

        Value src0 = valueMap.lookup(deInterleaveOp.getSrc0());
        Value src1 = valueMap.lookup(deInterleaveOp.getSrc1());
        if (!src0 || !src1)
            return failure();

        auto src0Type = dyn_cast<VectorType>(src0.getType());
        auto src1Type = dyn_cast<VectorType>(src1.getType());
        if (!src0Type || !src1Type || src0Type != src1Type)
            return failure();

        auto aveOp = b.create<hivmave::VFDeInterleaveOp>(loc, TypeRange{src0Type, src1Type}, ValueRange{src0, src1});

        valueMap[op.getResult(0)] = aveOp->getResult(0);
        valueMap[op.getResult(1)] = aveOp->getResult(1);
        return success();
    }

    // tla.gather: per-lane indexed load from a UB tile.
    //   x (tile_view → rank-1 memref) → VFGatherOp base
    //   y (loaded index vector)        → index_vec
    //   mask (optional)                → mask (all-true if absent)
    if (auto gatherOp = dyn_cast<::tla::GatherOp>(op)) {
        Value base = valueMap.lookup(gatherOp.getX());
        Value indexVec = valueMap.lookup(gatherOp.getY());
        if (!base || !indexVec)
            return failure();
        auto baseType = dyn_cast<MemRefType>(base.getType());
        if (!baseType || baseType.getRank() != 1)
            return failure();
        auto elemByteWidth = getElementByteWidth(baseType.getElementType());
        if (failed(elemByteWidth))
            return failure();
        int64_t numElems = 256 / *elemByteWidth;
        auto resultVecType = VectorType::get(numElems, baseType.getElementType());
        Value mask;
        if (gatherOp.getMask()) {
            mask = valueMap.lookup(gatherOp.getMask());
            if (!mask)
                return failure();
        } else {
            // Predicate follows the gathered vector semantic lane count.
            mask = fullMasks.get(loc, numElems);
        }
        Value zero = b.create<arith::ConstantIndexOp>(loc, 0);
        valueMap[gatherOp.getResult()] =
            b.create<hivmave::VFGatherOp>(loc, resultVecType, base, ValueRange(zero), indexVec, mask);
        return success();
    }

    if (auto info = getVectorUnaryInfo(&op)) {
        if (op.getNumResults() != 1)
            return failure();
        TlaUnaryOperands operands = info->operands;
        Value operand = valueMap.lookup(operands.operand);
        if (!operand)
            return failure();
        auto operandVecType = dyn_cast<VectorType>(operand.getType());
        if (!operandVecType)
            return failure();

        Type tlaOperandType = operands.operand.getType();
        if (isa<::tla::VectorSSAType>(tlaOperandType)) {
            if (failed(validateVectorUnaryElementType(&op, *info, operandVecType.getElementType())))
                return failure();
        } else if (!isa<::tla::MaskSSAType>(tlaOperandType)) {
            return op.emitError("expected !tla.vector<NxT> or !tla.mask<N> operand");
        }

        Value mask;
        if (operands.mask) {
            mask = valueMap.lookup(operands.mask);
            if (!mask)
                return failure();
        } else {
            mask = allTrueMaskFor(fullMasks, loc, operandVecType, tlaOperandType);
        }
        Value result = createVectorUnaryResult(b, loc, info->kind, tlaOperandType, operandVecType, operand, mask);
        if (!result)
            return failure();
        valueMap[op.getResult(0)] = result;
        return success();
    }

    // tla.create_mask: build a mask vector from a fixed pattern ->
    // ave.hir.pge<PATTERN>. The op's own dtype attr fixes the lane count
    // (256 bytes / element size) and hence the i1 mask width.
    if (auto maskOp = dyn_cast<::tla::CreateMaskOp>(op)) {
        auto pattern = hivmave::symbolizePgePattern(maskOp.getPattern());
        if (!pattern)
            return maskOp.emitError("unknown tla.create_mask pattern: ") << maskOp.getPattern(), failure();
        auto opCtx = deriveVecCtxForElement(maskOp.getDtype());
        if (failed(opCtx))
            return maskOp.emitError("unsupported tla.create_mask dtype: ") << maskOp.getDtype(), failure();
        valueMap[maskOp.getResult()] = createPredicatePge(b, loc, opCtx->maskVecType, opCtx->lanes, *pattern);
        return success();
    }

    // tla.update_mask: tail mask + remaining count. Lowers to ave.hir.plt,
    // whose mask result drives masked stores and whose second result
    // (true_shape - lanes) is threaded back as the loop-carried tail counter.
    // The op's own dtype attr fixes the lane count (256 bytes / element size)
    // and hence the i1 mask width and the tail decrement.
    if (auto updateMaskOp = dyn_cast<::tla::UpdateMaskOp>(op)) {
        // The true-shape operand may be a vec.func-external index constant (e.g.
        // tla.update_mask(1) building a single-lane mask): such constants are not
        // collected as helper arguments, so clone them inline like other scalar
        // operands instead of a bare valueMap lookup (which would miss them).
        Value trueShape = lookupOrCloneScalarValue(b, updateMaskOp.getTrueShape(), valueMap);
        if (!trueShape)
            return failure();
        auto opCtx = deriveVecCtxForElement(updateMaskOp.getDtype());
        if (failed(opCtx))
            return updateMaskOp.emitError("unsupported tla.update_mask dtype: ") << updateMaskOp.getDtype(), failure();
        auto plt = createPredicatePlt(b, loc, opCtx->maskVecType, opCtx->lanes, trueShape);
        valueMap[updateMaskOp.getMask()] = plt.getRes();
        // new_true_shape = true_shape - lanes, which is exactly what plt computes.
        // We materialize it with index arithmetic rather than consuming plt's second
        // result: that result is i32 in hardware but typed index, so carrying it
        // through the loop would leave an unfoldable i32<->index unrealized cast.
        Value lanesValue = b.create<arith::ConstantIndexOp>(loc, opCtx->lanes);
        valueMap[updateMaskOp.getNewTrueShape()] = b.create<arith::SubIOp>(loc, trueShape, lanesValue);
        return success();
    }

    if (auto cmpOp = dyn_cast<::tla::CmpOp>(op)) {
        Value lhs = valueMap.lookup(cmpOp.getLhs());
        if (!lhs)
            return failure();
        // The compare's operand width fixes both the input vectors and the i1 mask
        // result width.
        auto lhsTy = dyn_cast<VectorType>(lhs.getType());
        if (!lhsTy)
            return failure();
        auto opCtx = deriveVecCtxForElement(lhsTy.getElementType());
        if (failed(opCtx))
            return failure();
        auto cmpType = mapCmpMode(cmpOp.getMode());
        if (!cmpType)
            return cmpOp.emitError("unknown tla.cmp mode: ") << cmpOp.getMode(), failure();
        Value mask;
        if (cmpOp.getMask()) {
            mask = valueMap.lookup(cmpOp.getMask());
            if (!mask)
                return failure();
        } else {
            mask = fullMasks.get(loc, opCtx->lanes);
        }
        if (isa<::tla::VectorSSAType>(cmpOp.getRhs().getType())) {
            Value rhs = valueMap.lookup(cmpOp.getRhs());
            if (!rhs)
                return failure();
            valueMap[cmpOp.getResult()] = b.create<hivmave::VFCmpOp>(loc, opCtx->maskVecType, *cmpType, lhs, rhs, mask);
        } else {
            Value rhs = lookupOrCloneScalarValue(b, cmpOp.getRhs(), valueMap);
            if (!rhs)
                return failure();
            auto scalarOr = castScalarForVectorElement(rhs, opCtx->elementType);
            if (failed(scalarOr))
                return failure();
            valueMap[cmpOp.getResult()] =
                b.create<hivmave::VFCmpS>(loc, opCtx->maskVecType, *cmpType, lhs, *scalarOr, mask);
        }
        return success();
    }

    if (auto storeOp = dyn_cast<::tla::StoreOp>(op)) {
        Value dest = valueMap.lookup(storeOp.getDest());
        Value source = valueMap.lookup(storeOp.getSource());
        if (!dest || !source)
            return failure();

        // MaskSSA source: 1/2/4-byte UB ← i1 memref view ← masked_store <NORM_B8>.
        if (isa<::tla::MaskSSAType>(storeOp.getSource().getType())) {
            auto maskType = cast<::tla::MaskSSAType>(storeOp.getSource().getType());
            int64_t lanes = maskType.getPhysicalLanes();
            auto i1MemrefOr = materializeI1MaskMemrefFromUb(b, loc, dest, lanes);
            if (failed(i1MemrefOr))
                return storeOp.emitError("failed to materialize i1 memref view for tla.store MaskSSA"), failure();
            VectorType semanticMaskType = VectorType::get({lanes}, b.getI1Type());
            Value zero = b.create<arith::ConstantIndexOp>(loc, 0);
            Value allTrue = fullMasks.get(loc, lanes);
            Value storeVal = source;
            if (storeVal.getType() != semanticMaskType)
                storeVal = b.create<UnrealizedConversionCastOp>(loc, semanticMaskType, storeVal).getResult(0);
            Value storePred = allTrue;
            if (storePred.getType() != semanticMaskType)
                storePred = b.create<UnrealizedConversionCastOp>(loc, semanticMaskType, storePred).getResult(0);
            b.create<hivmave::VFMaskedStoreOp>(
                loc, hivmave::StoreDist::NORM_B8, *i1MemrefOr, ValueRange{zero}, storePred, storeVal);
            return success();
        }

        Value zero = b.create<arith::ConstantIndexOp>(loc, 0);
        Value mask;
        auto sourceTy = dyn_cast<VectorType>(source.getType());
        if (!sourceTy)
            return failure();
        auto opCtx = deriveVecCtxForElement(sourceTy.getElementType());
        if (failed(opCtx))
            return failure();
        if (storeOp.getMask()) {
            mask = valueMap.lookup(storeOp.getMask());
            if (!mask)
                return failure();
        } else {
            auto implicitMask = createImplicitVectorStoreMask(b, loc, storeOp, *opCtx, valueMap, fullMasks);
            if (failed(implicitMask))
                return storeOp.emitError("failed to determine omitted tla.store active lanes"), failure();
            mask = *implicitMask;
        }
        // store_unalign (this PR): mark AVE masked-store as unaligned UB access.
        if (storeOp.getUnalignedUbAccess().value_or(false)) {
            auto store = b.create<hivmave::VFMaskedStoreOp>(loc, dest, ValueRange{zero}, mask, source);
            store->setAttr(hivmave::UnalignedAttr::name, hivmave::UnalignedAttr::get(b.getContext()));
        } else if (storeOp.getBlockStride()) {
            Value blockStrideVal = lookupOrCloneScalarValue(b, storeOp.getBlockStride(), valueMap);
            if (!blockStrideVal)
                return failure();
            if (!isa<IntegerType>(blockStrideVal.getType()))
                blockStrideVal = b.create<arith::IndexCastOp>(loc, b.getI32Type(), blockStrideVal);
            std::string calleeName = getStoreWithStrideLibraryCallName(sourceTy.getElementType());
            if (calleeName.empty())
                return storeOp.emitError("unsupported element type for tla.store with BlockStoreParams: ")
                           << sourceTy.getElementType(),
                       failure();
            Value storeSrc = source;
            if (blockStrideUsesByteRegister(sourceTy.getElementType())) {
                storeSrc = b.create<mlir::vector::BitCastOp>(loc, byteRegisterTypeFor(sourceTy), source).getResult();
            }
            auto callee =
                getOrCreateStoreWithStrideLibraryCall(module, loc, storeSrc.getType(), dest.getType(), calleeName);
            if (!callee)
                return failure();
            VectorType pregVecType = fullPregVecType(b.getContext());
            Value pregMask = castMaskToPregType(b, loc, mask, pregVecType);
            b.create<func::CallOp>(loc, callee, ValueRange{storeSrc, dest, blockStrideVal, pregMask});
        } else {
            auto storeDistAttr = storeOp.getStoreDist();
            if (storeDistAttr && storeDistAttr->getStoreDist() != ::StoreDist::norm) {
                hivmave::StoreDist pattern = mapTlaStoreDistToAve(storeDistAttr->getStoreDist());
                bool isDual = isDualDestStoreDist(pattern);
                if (isDual) {
                    return storeOp.emitError("dual mode not implemented.");
                }
                b.create<hivmave::VFMaskedStoreOp>(loc, pattern, dest, ValueRange{zero}, mask, source);
            } else {
                // NORM mode
                b.create<hivmave::VFMaskedStoreOp>(loc, dest, ValueRange{zero}, mask, source);
            }
        }
        return success();
    }

    // scf.for: rebuild the loop, including loop-carried iter_args, and lower its
    // body. Init args and the scf.yield operands may be register, index, or
    // scalar SSA values threaded through the helper (e.g. the tail counter produced by tla.update_mask).
    if (auto forOp = dyn_cast<scf::ForOp>(op)) {
        Value lb = lookupOrCloneScalarValue(b, forOp.getLowerBound(), valueMap);
        Value ub = lookupOrCloneScalarValue(b, forOp.getUpperBound(), valueMap);
        Value step = lookupOrCloneScalarValue(b, forOp.getStep(), valueMap);
        if (!lb || !ub || !step)
            return failure();
        // Loop-carried `index` values (e.g. the tla.update_mask tail counter) are
        // carried across the loop as i64 instead: after scf->cf lowering the
        // downstream index->iN conversion only rewrites the induction variable, so
        // an index iter_arg would leave dangling index<->iN unrealized casts on the
        // carried value that ReconcileUnrealizedCasts cannot fold across the cf
        // block boundary. Casting at the boundaries with arith.index_cast keeps the
        // carried value a plain integer that lowers cleanly.
        Type i64Ty = b.getIntegerType(64);
        auto regionIterArgs = forOp.getRegionIterArgs();
        SmallVector<bool> wasIndex(regionIterArgs.size(), false);
        SmallVector<Value> initArgs;
        for (auto [idx, init] : llvm::enumerate(forOp.getInitArgs())) {
            Value mapped = lookupOrCloneScalarValue(b, init, valueMap);
            if (!mapped)
                return failure();
            if (isa<IndexType>(mapped.getType())) {
                wasIndex[idx] = true;
                mapped = b.create<arith::IndexCastOp>(loc, i64Ty, mapped);
            }
            initArgs.push_back(mapped);
        }
        LogicalResult bodyStatus = success();
        auto newFor = b.create<scf::ForOp>(
            loc, lb, ub, step, initArgs, [&](OpBuilder& nb, Location nloc, Value iv, ValueRange iterArgs) {
                DenseMap<Value, Value> nestedMap = valueMap;
                nestedMap[forOp.getInductionVar()] = iv;
                for (size_t i = 0; i < regionIterArgs.size(); ++i) {
                    Value newArg = iterArgs[i];
                    if (wasIndex[i])
                        newArg = nb.create<arith::IndexCastOp>(nloc, nb.getIndexType(), newArg);
                    nestedMap[regionIterArgs[i]] = newArg;
                }
                if (failed(lowerNestedVectorBlock(forOp.getBody(), nb, module, nestedMap, fullMasks))) {
                    bodyStatus = failure();
                    nb.create<scf::YieldOp>(nloc, iterArgs);
                    return;
                }
                auto oldYield = cast<scf::YieldOp>(forOp.getBody()->getTerminator());
                SmallVector<Value> yielded;
                for (auto [i, v] : llvm::enumerate(oldYield.getOperands())) {
                    Value mapped = lookupOrCloneScalarValue(nb, v, nestedMap);
                    if (!mapped) {
                        bodyStatus = failure();
                        break;
                    }
                    if (wasIndex[i] && isa<IndexType>(mapped.getType()))
                        mapped = nb.create<arith::IndexCastOp>(nloc, i64Ty, mapped);
                    yielded.push_back(mapped);
                }
                if (failed(bodyStatus)) {
                    nb.create<scf::YieldOp>(nloc, iterArgs);
                    return;
                }
                nb.create<scf::YieldOp>(nloc, yielded);
            });
        if (failed(bodyStatus))
            return failure();
        for (auto [i, oldRes] : llvm::enumerate(forOp.getResults())) {
            Value newRes = newFor.getResult(i);
            if (wasIndex[i] && !oldRes.use_empty())
                newRes = b.create<arith::IndexCastOp>(loc, b.getIndexType(), newRes);
            valueMap[oldRes] = newRes;
        }
        return success();
    }

    // scf.if: rebuild result-bearing conditionals after converting register
    // carrier types to physical builtin vectors. Each branch is lowered with an
    // independent value map, and its old scf.yield operands feed the new op.
    if (auto ifOp = dyn_cast<scf::IfOp>(op)) {
        Value cond = lookupOrCloneScalarValue(b, ifOp.getCondition(), valueMap);
        if (!cond)
            return failure();

        SmallVector<Type> resultTypes;
        for (Value result : ifOp.getResults()) {
            auto loweredType = lowerSCFCarrierType(result.getType());
            if (failed(loweredType))
                return failure();
            resultTypes.push_back(*loweredType);
        }

        bool hasElse = !ifOp.getElseRegion().empty();
        auto newIf = b.create<scf::IfOp>(loc, resultTypes, cond, hasElse);
        auto lowerBranch = [&](Block* oldBlock, Block* newBlock) -> LogicalResult {
            DenseMap<Value, Value> branchMap = valueMap;
            if (!newBlock->empty() && newBlock->back().hasTrait<OpTrait::IsTerminator>())
                newBlock->back().erase();
            OpBuilder branchBuilder = OpBuilder::atBlockEnd(newBlock);
            if (failed(lowerNestedVectorBlock(oldBlock, branchBuilder, module, branchMap, fullMasks)))
                return failure();

            auto oldYield = dyn_cast<scf::YieldOp>(oldBlock->getTerminator());
            if (!oldYield)
                return failure();
            SmallVector<Value> yielded;
            for (Value operand : oldYield.getOperands()) {
                Value mapped = lookupOrCloneScalarValue(branchBuilder, operand, branchMap);
                if (!mapped)
                    return failure();
                yielded.push_back(mapped);
            }
            branchBuilder.create<scf::YieldOp>(oldYield.getLoc(), yielded);
            return success();
        };

        if (failed(lowerBranch(ifOp.thenBlock(), newIf.thenBlock())))
            return failure();
        if (hasElse && failed(lowerBranch(ifOp.elseBlock(), newIf.elseBlock())))
            return failure();

        for (auto [oldResult, newResult] : llvm::zip(ifOp.getResults(), newIf.getResults()))
            valueMap[oldResult] = newResult;
        return success();
    }

    // Index/scalar arithmetic (arith.*) feeding offsets/conditions: clone with
    // mapped operands.
    if (op.getDialect()->getNamespace() == arith::ArithDialect::getDialectNamespace()) {
        IRMapping mapper;
        for (Value operand : op.getOperands()) {
            Value mapped = lookupOrCloneScalarValue(b, operand, valueMap);
            if (!mapped)
                return failure();
            mapper.map(operand, mapped);
        }
        Operation* cloned = b.clone(op, mapper);
        for (auto [oldResult, newResult] : llvm::zip(op.getResults(), cloned->getResults()))
            valueMap[oldResult] = newResult;
        return success();
    }

    // GM/UB scalar accesses lowered by tla-lower-scalar-access before outlining.
    if (auto memLoad = dyn_cast<mlir::memref::LoadOp>(op)) {
        Value mem = valueMap.lookup(memLoad.getMemRef());
        if (!mem)
            return failure();
        SmallVector<Value, 2> indices;
        for (Value idx : memLoad.getIndices()) {
            Value mapped = lookupOrCloneScalarValue(b, idx, valueMap);
            if (!mapped)
                return failure();
            indices.push_back(mapped);
        }
        valueMap[memLoad.getResult()] = b.create<mlir::memref::LoadOp>(loc, mem, indices).getResult();
        return success();
    }
    if (auto memStore = dyn_cast<mlir::memref::StoreOp>(op)) {
        Value mem = valueMap.lookup(memStore.getMemRef());
        Value val = valueMap.lookup(memStore.getValue());
        if (!val)
            val = lookupOrCloneScalarValue(b, memStore.getValue(), valueMap);
        if (!mem || !val)
            return failure();
        SmallVector<Value, 2> indices;
        for (Value idx : memStore.getIndices()) {
            Value mapped = lookupOrCloneScalarValue(b, idx, valueMap);
            if (!mapped)
                return failure();
            indices.push_back(mapped);
        }
        b.create<mlir::memref::StoreOp>(loc, val, mem, indices);
        return success();
    }

    // Rebuild the descriptor view used by a scalar access from the base memref
    // passed to the helper.
    if (isa<mlir::memref::ExtractStridedMetadataOp, mlir::memref::ReinterpretCastOp>(op)) {
        IRMapping mapper;
        for (Value operand : op.getOperands()) {
            Value mapped = lookupOrCloneScalarValue(b, operand, valueMap);
            if (!mapped)
                return failure();
            mapper.map(operand, mapped);
        }
        Operation* cloned = b.clone(op, mapper);
        for (auto [oldResult, newResult] : llvm::zip(op.getResults(), cloned->getResults()))
            valueMap[oldResult] = newResult;
        return success();
    }

    // Index/scalar arithmetic (arith.*) feeding tensor_desc offset/stride operands:
    // clone with operands remapped. tla-lower-tensor-desc emits the coord/origin
    // arithmetic (addi/subi/minsi) for dynamic-coord tile_views; static cases fold
    // to constants (createOrFold) and are handled by the arith::ConstantOp arm above.
    if (op.getDialect()->getNamespace() == arith::ArithDialect::getDialectNamespace()) {
        IRMapping mapper;
        for (Value operand : op.getOperands()) {
            Value mapped = lookupOrCloneScalarValue(b, operand, valueMap);
            if (!mapped)
                return failure();
            mapper.map(operand, mapped);
        }
        Operation* cloned = b.clone(op, mapper);
        for (auto [oldResult, newResult] : llvm::zip(op.getResults(), cloned->getResults()))
            valueMap[oldResult] = newResult;
        return success();
    }

    if (op.hasTrait<OpTrait::IsTerminator>())
        return success();

    return failure();
}

static LogicalResult lowerNestedVectorBlock(
    Block* sourceBlock, OpBuilder& b, ModuleOp module, DenseMap<Value, Value>& valueMap,
    AutoFullPredicateCache& fullMasks)
{
    for (Operation& op : sourceBlock->getOperations()) {
        // Terminators are reproduced by the enclosing op (scf.for/scf.if) or by
        // buildHelperFunc's func.return.
        if (op.hasTrait<OpTrait::IsTerminator>())
            continue;
        if (failed(lowerNestedVectorOp(op, b, module, valueMap, fullMasks)))
            return failure();
    }
    return success();
}

static Value stripHelperAddressCasts(Value address)
{
    while (Operation* def = address.getDefiningOp()) {
        if (!isa<arith::IndexCastOp, arith::ExtSIOp, arith::TruncIOp>(def))
            break;
        address = def->getOperand(0);
    }
    return address;
}

// Return the allocation identity used to deduplicate tensor descriptors and
// scalar-access memrefs that point at the same storage.
static Value getVectorHelperStorageIdentity(Value source)
{
    if (auto descOp = source.getDefiningOp<::tla::TensorDescOp>()) {
        source = descOp.getBase();
        if (auto intToPtr = source.getDefiningOp<::tla::IntToPtrOp>())
            return stripHelperAddressCasts(intToPtr.getAddr());
    }
    while (auto castOp = source.getDefiningOp<mlir::memref::CastOp>())
        source = castOp.getSource();
    if (auto pointerCast = source.getDefiningOp<hivm::PointerCastOp>())
        return stripHelperAddressCasts(pointerCast.getSingleAddr());
    return source;
}

static FailureOr<MemRefType> getVectorHelperOperandType(Value source)
{
    if (auto type = dyn_cast<MemRefType>(source.getType()))
        return type;
    return getVectorHelperArgMemrefType(source);
}

// Add `source` as a helper operand unless another operand already represents
// the same allocation with the same helper memref type. Record every source's
// representative so scalar views can reuse a tensor operand's helper argument.
static void addVectorHelperOperand(Value source, SmallVectorImpl<Value>& operands, DenseMap<Value, Value>& aliases)
{
    Value identity = getVectorHelperStorageIdentity(source);
    FailureOr<MemRefType> sourceType = getVectorHelperOperandType(source);
    for (Value operand : operands) {
        FailureOr<MemRefType> operandType = getVectorHelperOperandType(operand);
        if (succeeded(sourceType) && succeeded(operandType) && *sourceType == *operandType &&
            identity == getVectorHelperStorageIdentity(operand)) {
            aliases[source] = operand;
            return;
        }
    }
    operands.push_back(source);
    aliases[source] = source;
}

// Scalar access lowering builds extract_strided_metadata + reinterpret_cast
// inside vec.func. Pass the first memref defined outside the outlined region to
// the helper and rebuild that view inside the helper.
static FailureOr<Value> getScalarAccessHelperMemref(
    Value source, Region& outlinedRegion, Operation* callSite, DominanceInfo& dominanceInfo)
{
    while (Operation* def = source.getDefiningOp()) {
        Region* defRegion = source.getParentRegion();
        if (!defRegion || !outlinedRegion.isAncestor(defRegion))
            break;
        if (!isa<mlir::memref::ExtractStridedMetadataOp, mlir::memref::ReinterpretCastOp>(def))
            break;
        source = def->getOperand(0);
    }

    // A dynamic inttoptr descriptor can materialize its hivm.pointer_cast at the
    // scalar access insertion point, which may be inside vec.func. Such a value
    // cannot be passed to the helper call that replaces the enclosing vec.func.
    if (!dominanceInfo.dominates(source, callSite))
        return failure();
    return source;
}

// Collect, in body order, the unique base memrefs that tla.load/tla.store/
// tla.gather chunks and bridged scalar accesses reference. These become the
// helper's arguments.
static LogicalResult collectVectorHelperOperands(
    Block* block, Region& outlinedRegion, Operation* callSite, DominanceInfo& dominanceInfo,
    SmallVectorImpl<Value>& operands, DenseMap<Value, Value>& aliases)
{
    for (Operation& op : block->getOperations()) {
        if (auto loadOp = dyn_cast<::tla::LoadOp>(op)) {
            addVectorHelperOperand(loadOp.getSource(), operands, aliases);
            continue;
        }
        if (auto storeOp = dyn_cast<::tla::StoreOp>(op)) {
            addVectorHelperOperand(storeOp.getDest(), operands, aliases);
            continue;
        }
        if (auto gatherOp = dyn_cast<::tla::GatherOp>(op)) {
            addVectorHelperOperand(gatherOp.getX(), operands, aliases);
            continue;
        }
        // Bridged scalar accesses (after tla-lower-scalar-access) appear as
        // memref.load/store.
        if (auto memLoad = dyn_cast<mlir::memref::LoadOp>(op)) {
            FailureOr<Value> source =
                getScalarAccessHelperMemref(memLoad.getMemRef(), outlinedRegion, callSite, dominanceInfo);
            if (failed(source))
                return failure();
            addVectorHelperOperand(*source, operands, aliases);
            continue;
        }
        if (auto memStore = dyn_cast<mlir::memref::StoreOp>(op)) {
            FailureOr<Value> source =
                getScalarAccessHelperMemref(memStore.getMemRef(), outlinedRegion, callSite, dominanceInfo);
            if (failed(source))
                return failure();
            addVectorHelperOperand(*source, operands, aliases);
            continue;
        }
        for (Region& region : op.getRegions())
            for (Block& nested : region)
                if (failed(collectVectorHelperOperands(
                        &nested, outlinedRegion, callSite, dominanceInfo, operands, aliases)))
                    return failure();
    }
    return success();
}

// Collect unique scalar values used inside the region but defined outside it
// (e.g. a sub_block_idx/block_idx computed at the top of the kernel, or a
// vector-scalar RHS constant). Passing them into the helper avoids cloning float
// constants into vector helpers where vector.broadcast can fold to illegal
// vector arith.constant ops before the HIVMAVE conversion pipeline.
// They are passed in as trailing scalar arguments rather than recomputed inside
// the outlined vector function.
static void collectVectorHelperScalarOperands(::tla::VecFuncOp vecFuncOp, SmallVectorImpl<Value>& scalars)
{
    vecFuncOp.walk([&](Operation* op) {
        for (Value operand : op->getOperands()) {
            Type operandType = operand.getType();
            if (!operandType.isIntOrIndex() && !isa<FloatType>(operandType))
                continue;
            // Index/integer constants defined outside vec.func are cloned inline by
            // lookupOrCloneScalarValue rather than passed as helper args. This keeps
            // static tile offsets/strides (now tensor_desc operands, materialized at
            // function scope by tla-lower-tensor-desc) as constants inside the helper
            // instead of bloating the helper signature. Float constants are still
            // passed as args (see the comment above re: vector.broadcast folding).
            if (operandType.isIntOrIndex() && operand.getDefiningOp<arith::ConstantOp>())
                continue;
            Region* defRegion = operand.getParentRegion();
            if (defRegion && !vecFuncOp.getBody().isAncestor(defRegion) && !llvm::is_contained(scalars, operand))
                scalars.push_back(operand);
        }
    });
    // The walk above visits tensor_descs written inside vec.func. A descriptor
    // produced in the parent scope and merely *used* here keeps its offset and
    // stride operands on an op the walk never reaches, yet buildHelperFunc has to
    // materialize that descriptor inside the helper. Capture those operands too,
    // under the same rule: constants are cloned inline, and anything else becomes
    // a trailing helper argument.
    SmallVector<::tla::TensorDescOp> externalDescs;
    collectExternalTensorDescs(vecFuncOp, externalDescs);
    for (::tla::TensorDescOp desc : externalDescs) {
        for (Value operand : desc->getOperands()) {
            Type operandType = operand.getType();
            if (!operandType.isIntOrIndex() && !isa<FloatType>(operandType))
                continue;
            if (operandType.isIntOrIndex() && operand.getDefiningOp<arith::ConstantOp>())
                continue;
            if (!llvm::is_contained(scalars, operand))
                scalars.push_back(operand);
        }
    }
}

// Build a vector_region helper for a tla.vec.func body. The helper receives one
// flat on-chip memref per referenced tensor (or a concrete memref for scalar
// accesses); the for/if control flow is carried inside the helper, where each
// tla.load/store is lowered to an AVE vload/masked-store over a 256-byte tile
// carved from the base address at the per-iteration offset.
static FailureOr<func::FuncOp> buildHelperFunc(
    ModuleOp module, func::FuncOp parentFunc, ::tla::VecFuncOp vecFuncOp, ArrayRef<Value> helperOperands,
    const DenseMap<Value, Value>& helperOperandAliases, ArrayRef<Value> scalarOperands, int& nextVectorRegionId,
    DenseMap<Value, Value>& loweredMemrefByValue)
{
    MLIRContext* ctx = module.getContext();
    Operation* vectorOp = vecFuncOp.getOperation();
    OpBuilder moduleBuilder(module.getBodyRegion());
    moduleBuilder.setInsertionPointAfter(parentFunc);

    Block* body = vecFuncOp.getBody().empty() ? nullptr : &vecFuncOp.getBody().front();
    if (!body || helperOperands.empty())
        return failure();

    SmallVector<Type> functionInputs;
    functionInputs.reserve(helperOperands.size());
    for (Value operand : helperOperands) {
        // Already-bridged scalar-access memrefs keep their concrete type.
        if (auto mt = dyn_cast<MemRefType>(operand.getType())) {
            functionInputs.push_back(mt);
            continue;
        }
        auto operandType = getVectorHelperArgMemrefType(operand);
        if (failed(operandType))
            return failure();
        functionInputs.push_back(*operandType);
    }
    // Trailing scalar args: scalars captured from outside the region.
    for (Value scalar : scalarOperands)
        functionInputs.push_back(scalar.getType());
    auto funcType = moduleBuilder.getFunctionType(functionInputs, TypeRange{});

    // The per-iteration vector tile is one 256-byte register's worth of elements.
    // Each op inside the helper derives its own width from its operands/result,
    // so tiles may carry different element types within one region (e.g. a f32
    // load feeding a tla.cast to f16). Validate only that each tile operand is a
    // supported int/float type (the trailing scalar args are index/int and are
    // handled separately). This runs before the helper is created so a validation
    // failure leaks no partial IR.
    for (size_t i = 0; i < helperOperands.size(); ++i) {
        Type tileElementType = cast<MemRefType>(functionInputs[i]).getElementType();
        if (!isa<IntegerType>(tileElementType) && !isa<FloatType>(tileElementType))
            return vectorOp->emitError("unsupported element type for vector binary helper: ") << tileElementType;
        if (failed(getVectorLaneCount(tileElementType)))
            return vectorOp->emitError("unsupported element width for vector helper tile: ") << tileElementType;
    }

    std::string helperName = buildUniqueVectorHelperName(module, nextVectorRegionId);
    auto helper = moduleBuilder.create<func::FuncOp>(vectorOp->getLoc(), helperName, funcType);
    helper.setPrivate();
    helper->setAttr(hivm::TFuncCoreTypeAttr::name, hivm::TFuncCoreTypeAttr::get(ctx, hivm::TFuncCoreType::AIV));
    helper->setAttr("hivm.vector_function", UnitAttr::get(ctx));
    helper->setAttr("no_inline", UnitAttr::get(ctx));

    Block* entry = helper.addEntryBlock();
    OpBuilder b = OpBuilder::atBlockBegin(entry);

    DenseMap<Value, Value> valueMap;
    for (auto [i, operand] : llvm::enumerate(helperOperands))
        valueMap[operand] = entry->getArgument(i);
    for (auto [source, representative] : helperOperandAliases)
        if (Value argument = valueMap.lookup(representative))
            valueMap[source] = argument;
    // Captured scalars map to their trailing block arguments.
    for (auto [j, scalar] : llvm::enumerate(scalarOperands))
        valueMap[scalar] = entry->getArgument(helperOperands.size() + j);
    // helperOperands holds one representative tensor_desc per unique base (see
    // addVectorHelperOperand). Every other vec.func-internal tensor_desc sharing
    // that base must resolve to the same helper memref arg so the TensorDescOp arm
    // in lowerNestedVectorOp can map it.
    DenseMap<Value, Value> baseToArg;
    for (Value operand : helperOperands)
        if (auto d = operand.getDefiningOp<::tla::TensorDescOp>())
            baseToArg[d.getBase()] = valueMap[operand];
    vecFuncOp.walk([&](::tla::TensorDescOp desc) {
        if (auto it = baseToArg.find(desc.getBase()); it != baseToArg.end())
            valueMap[desc.getResult()] = it->second;
    });
    // A descriptor produced *outside* the region and used inside it needs the
    // same treatment as an internal one. lowerNestedVectorBlock never walks it
    // (the op lives in the parent), so without this its value would stay mapped
    // to the bare base-memref argument seeded above and every access would
    // silently address base + 0 -- the descriptor's offset lost.
    // addVectorHelperOperand keys operands by storage identity, so several
    // descriptors over one allocation collapse to a single argument and the
    // offset cannot come from the ABI; materialize it here instead, after
    // baseToArg has recorded the unoffset arguments.
    SmallVector<::tla::TensorDescOp> externalDescs;
    collectExternalTensorDescs(vecFuncOp, externalDescs);
    for (::tla::TensorDescOp desc : externalDescs) {
        // Only descriptors whose base reached the helper as an argument: the rest
        // are not addressable from in here and are not referenced by the body.
        if (!valueMap.count(desc.getResult()))
            continue;
        // A non-constant coord or stride cannot be reached from here (it is not a
        // helper argument), so this reports the unsupported descriptor instead of
        // falling back to offset 0.
        if (failed(materializeTensorDescSubview(desc, b, valueMap))) {
            helper.erase();
            return failure();
        }
    }

    AutoFullPredicateCache fullMasks(&helper.getBody().front());
    if (failed(lowerNestedVectorBlock(body, b, module, valueMap, fullMasks))) {
        // Discard the partially-built helper so an unsupported construct fails
        // cleanly (the vec.func is left intact) instead of leaking malformed IR.
        helper.erase();
        return failure();
    }
    b.create<func::ReturnOp>(vectorOp->getLoc());
    return helper;
}

// tla.vec.func carries its execution model in a "mode" string attribute
// ("simd" / "simt", either case). SIMD is the historical default and is assumed
// when the attribute is absent.
static StringRef getVecFuncMode(::tla::VecFuncOp vecFuncOp)
{
    auto modeAttr = vecFuncOp->getAttrOfType<StringAttr>("mode");
    return modeAttr ? modeAttr.getValue() : StringRef("simd");
}

static bool isSimtVecFunc(::tla::VecFuncOp vecFuncOp)
{
    return getVecFuncMode(vecFuncOp).equals_insensitive("simt");
}

static bool isSimdVecFunc(::tla::VecFuncOp vecFuncOp)
{
    return !isSimtVecFunc(vecFuncOp);
}

// Lower a TLA op with three i32 results onto three separate regbase intrinsics,
// one per component. The intrinsics are modelled as independent nullary ops
// (LLVM intrinsics cannot return an aggregate), so a result nobody reads simply
// never gets an intrinsic emitted for it -- `tid, _, _ = tla.arch.thread_idx()`
// costs exactly one instruction.
template <typename TlaOpTy, typename XOpTy, typename YOpTy, typename ZOpTy>
class LowerSimtTripleOpPattern : public OpRewritePattern<TlaOpTy> {
public:
    using OpRewritePattern<TlaOpTy>::OpRewritePattern;

    LogicalResult matchAndRewrite(TlaOpTy op, PatternRewriter& rewriter) const override
    {
        Location loc = op.getLoc();
        Type i32 = rewriter.getI32Type();

        if (op.getX().use_empty() && op.getY().use_empty() && op.getZ().use_empty()) {
            rewriter.eraseOp(op);
            return success();
        }

        // Replace uses individually rather than via replaceOp: that way an unused
        // component needs no placeholder value and no intrinsic.
        if (!op.getX().use_empty())
            rewriter.replaceAllUsesWith(op.getX(), rewriter.create<XOpTy>(loc, i32).getRes());
        if (!op.getY().use_empty())
            rewriter.replaceAllUsesWith(op.getY(), rewriter.create<YOpTy>(loc, i32).getRes());
        if (!op.getZ().use_empty())
            rewriter.replaceAllUsesWith(op.getZ(), rewriter.create<ZOpTy>(loc, i32).getRes());
        rewriter.eraseOp(op);
        return success();
    }
};

using LowerThreadIdxPattern = LowerSimtTripleOpPattern<
    ::tla::ThreadIdxOp, hivm_regbaseintrins::ThreadIdXOp, hivm_regbaseintrins::ThreadIdYOp,
    hivm_regbaseintrins::ThreadIdZOp>;

using LowerBlockDimPattern = LowerSimtTripleOpPattern<
    ::tla::ThreadBlockDimOp, hivm_regbaseintrins::BlockDimXOp, hivm_regbaseintrins::BlockDimYOp,
    hivm_regbaseintrins::BlockDimZOp>;

// ---------------------------------------------------------------------------
// SIMT vec.func outlining
//
// A SIMT region is per-thread scalar code, not whole-vector AVE work, so it is
// outlined into its own function and invoked through
// hivm_regbaseintrins.intrins.launch_func rather than being folded into the
// enclosing AIV body. The outlined function must stay in the plain
// memref/scf/arith form that hivmc-a5 accepts for a simt_entry function -- no
// reinterpret_cast views, no descriptor arithmetic. Everything the body needs is
// therefore materialized *before* the launch and handed over as arguments:
// buffers as raw pointers (extract_aligned_pointer_as_index -> index_cast ->
// llvm.inttoptr), thread geometry as the launch's thread_block_dim triple.
// ---------------------------------------------------------------------------

// Values produced outside `region` but used inside it. Constant-like producers
// are reported separately: they are cloned into the outlined body instead of
// consuming an ABI slot.
struct SimtCaptures {
    SmallVector<Value, 4> tensors;        // tla.tensor values reached by scalar access
    SmallVector<Value, 4> scalars;        // everything else that must be passed in
    SmallVector<Operation*, 8> constants; // clone into the body
};

static bool isDefinedOutside(Value value, Region& region)
{
    Region* defRegion = value.getParentRegion();
    return !defRegion || !region.isAncestor(defRegion);
}

static void collectSimtCaptures(::tla::VecFuncOp vecFuncOp, SimtCaptures& captures)
{
    Region& region = vecFuncOp.getBody();
    SetVector<Value> tensorSet;
    SetVector<Value> scalarSet;
    SetVector<Operation*> constantSet;

    region.walk([&](Operation* op) {
        // Tensor operands of scalar accesses become memref parameters.
        if (auto load = dyn_cast<::tla::SimtLoadOp>(op)) {
            if (isDefinedOutside(load.getSource(), region))
                tensorSet.insert(load.getSource());
        } else if (auto store = dyn_cast<::tla::SimtStoreOp>(op)) {
            if (isDefinedOutside(store.getDest(), region))
                tensorSet.insert(store.getDest());
        }
        for (Value operand : op->getOperands()) {
            if (!isDefinedOutside(operand, region))
                continue;
            if (tensorSet.contains(operand) || isa<::tla::TlaTensorType>(operand.getType()))
                continue;
            if (Operation* def = operand.getDefiningOp()) {
                if (def->hasTrait<mlir::OpTrait::ConstantLike>()) {
                    constantSet.insert(def);
                    continue;
                }
            }
            scalarSet.insert(operand);
        }
    });

    captures.tensors.assign(tensorSet.begin(), tensorSet.end());
    captures.scalars.assign(scalarSet.begin(), scalarSet.end());
    captures.constants.assign(constantSet.begin(), constantSet.end());
}

// Same fan-out as LowerSimtTripleOpPattern, applied directly to a finished
// function. The outlining clones the region before the rewrite driver reaches
// its ops, so the identity queries have to be lowered here rather than left to
// the pattern -- the outlined function is not re-driven.
template <typename TlaOpTy, typename XOpTy, typename YOpTy, typename ZOpTy>
static void lowerSimtTripleOpsIn(func::FuncOp vf)
{
    SmallVector<TlaOpTy, 4> ops;
    vf.walk([&](TlaOpTy op) { ops.push_back(op); });
    for (TlaOpTy op : ops) {
        OpBuilder builder(op);
        Location loc = op.getLoc();
        Type i32 = builder.getI32Type();
        if (!op.getX().use_empty())
            op.getX().replaceAllUsesWith(builder.create<XOpTy>(loc, i32).getRes());
        if (!op.getY().use_empty())
            op.getY().replaceAllUsesWith(builder.create<YOpTy>(loc, i32).getRes());
        if (!op.getZ().use_empty())
            op.getZ().replaceAllUsesWith(builder.create<ZOpTy>(loc, i32).getRes());
        op.erase();
    }
}

// Address spaces the SIMT ABI addresses device buffers through, on both the
// launch's !llvm.ptr operands and the outlined function's memref parameters.
// The two must agree or hivmc-a5 rejects the call signature. These are the
// integer forms: #hivm.address_space<...> lowers to a descriptor struct rather
// than a bare pointer, which does not match what launch_func passes.
static constexpr int kSimtGmAddressSpace = 1;
static constexpr int kSimtUbAddressSpace = 6;

// Integer address space for a descriptor's address space name, or failure for
// one a SIMT vector function cannot reach.
static std::optional<int> simtAddressSpaceFor(StringRef addrspace)
{
    if (addrspace == "gm")
        return kSimtGmAddressSpace;
    if (addrspace == "ub")
        return kSimtUbAddressSpace;
    return std::nullopt;
}

// The per-thread scalar arithmetic ops carry their own identity through the TLA
// pipeline so a SIMT body reads as TLA IR rather than raw arith; here each one
// becomes the arith op for its element type. Integer division uses the signed
// form: the frontend only emits tla.simt_div for floats (Numeric '/' rejects
// integers), so an integer divide can only arrive from hand-written TLA IR.
// tla.simt_cast -> the arith conversion its `kind` names.
static void lowerSimtCastIn(func::FuncOp vf)
{
    SmallVector<::tla::SimtCastOp, 8> ops;
    vf.walk([&](::tla::SimtCastOp op) { ops.push_back(op); });
    for (::tla::SimtCastOp op : ops) {
        OpBuilder builder(op);
        Location loc = op.getLoc();
        Type resultType = op.getResult().getType();
        Value src = op.getSource();
        StringRef kind = op.getKind();
        Value lowered;
        if (kind == "extsi")
            lowered = builder.create<arith::ExtSIOp>(loc, resultType, src);
        else if (kind == "extui")
            lowered = builder.create<arith::ExtUIOp>(loc, resultType, src);
        else if (kind == "trunci")
            lowered = builder.create<arith::TruncIOp>(loc, resultType, src);
        else if (kind == "extf")
            // llvm.fpext / llvm.fptrunc rather than the arith ops: for bf16
            // convert-hivm-to-std rewrites those into vcast_*_1d_with_mode
            // vector-template calls that the c310 bitcode does not export.
            lowered = builder.create<LLVM::FPExtOp>(loc, resultType, src);
        else if (kind == "truncf")
            lowered = builder.create<LLVM::FPTruncOp>(loc, resultType, src);
        else if (kind == "sitofp")
            lowered = builder.create<arith::SIToFPOp>(loc, resultType, src);
        else if (kind == "uitofp")
            lowered = builder.create<arith::UIToFPOp>(loc, resultType, src);
        else if (kind == "fptosi")
            lowered = builder.create<arith::FPToSIOp>(loc, resultType, src);
        else
            lowered = builder.create<arith::FPToUIOp>(loc, resultType, src);
        op.getResult().replaceAllUsesWith(lowered);
        op.erase();
    }
}

// tla.simt_where -> arith.select. The condition is already an i1, so the
// operands carry the result type and the plain builder suffices.
static void lowerSimtWhereIn(func::FuncOp vf)
{
    SmallVector<::tla::SimtWhereOp, 8> ops;
    vf.walk([&](::tla::SimtWhereOp op) { ops.push_back(op); });
    for (::tla::SimtWhereOp op : ops) {
        OpBuilder builder(op);
        Value lowered =
            builder.create<arith::SelectOp>(op.getLoc(), op.getCondition(), op.getX(), op.getY()).getResult();
        op.getResult().replaceAllUsesWith(lowered);
        op.erase();
    }
}

// tla.simt_cmp -> arith.cmpf (ordered) / arith.cmpi, picking the signed or
// unsigned integer predicate the frontend recorded.
static void lowerSimtCmpIn(func::FuncOp vf)
{
    SmallVector<::tla::SimtCmpOp, 8> ops;
    vf.walk([&](::tla::SimtCmpOp op) { ops.push_back(op); });
    for (::tla::SimtCmpOp op : ops) {
        OpBuilder builder(op);
        Location loc = op.getLoc();
        StringRef mode = op.getMode();
        Value lowered;
        if (isa<FloatType>(op.getLhs().getType())) {
            arith::CmpFPredicate pred = llvm::StringSwitch<arith::CmpFPredicate>(mode)
                                            .Case("lt", arith::CmpFPredicate::OLT)
                                            .Case("le", arith::CmpFPredicate::OLE)
                                            .Case("gt", arith::CmpFPredicate::OGT)
                                            .Case("ge", arith::CmpFPredicate::OGE)
                                            .Case("eq", arith::CmpFPredicate::OEQ)
                                            .Default(arith::CmpFPredicate::ONE);
            lowered = builder.create<arith::CmpFOp>(loc, pred, op.getLhs(), op.getRhs()).getResult();
        } else {
            bool isUnsigned = op.getIsUnsigned();
            arith::CmpIPredicate pred =
                llvm::StringSwitch<arith::CmpIPredicate>(mode)
                    .Case("lt", isUnsigned ? arith::CmpIPredicate::ult : arith::CmpIPredicate::slt)
                    .Case("le", isUnsigned ? arith::CmpIPredicate::ule : arith::CmpIPredicate::sle)
                    .Case("gt", isUnsigned ? arith::CmpIPredicate::ugt : arith::CmpIPredicate::sgt)
                    .Case("ge", isUnsigned ? arith::CmpIPredicate::uge : arith::CmpIPredicate::sge)
                    .Case("eq", arith::CmpIPredicate::eq)
                    .Default(arith::CmpIPredicate::ne);
            lowered = builder.create<arith::CmpIOp>(loc, pred, op.getLhs(), op.getRhs()).getResult();
        }
        op.getResult().replaceAllUsesWith(lowered);
        op.erase();
    }
}

// Same as lowerSimtBinaryIn, but for LLVM intrinsic ops, whose builders need
// the result type spelled out.
template <typename TlaOpTy, typename FloatOpTy, typename IntOpTy>
static void lowerSimtIntrinBinaryIn(func::FuncOp vf)
{
    SmallVector<TlaOpTy, 8> ops;
    vf.walk([&](TlaOpTy op) { ops.push_back(op); });
    for (TlaOpTy op : ops) {
        OpBuilder builder(op);
        Location loc = op.getLoc();
        Type resultType = op.getResult().getType();
        Value lowered = isa<FloatType>(resultType) ?
                            builder.create<FloatOpTy>(loc, resultType, op.getLhs(), op.getRhs()).getResult() :
                            builder.create<IntOpTy>(loc, resultType, op.getLhs(), op.getRhs()).getResult();
        op.getResult().replaceAllUsesWith(lowered);
        op.erase();
    }
}

template <typename TlaOpTy, typename FloatOpTy, typename IntOpTy>
static void lowerSimtBinaryIn(func::FuncOp vf)
{
    SmallVector<TlaOpTy, 8> ops;
    vf.walk([&](TlaOpTy op) { ops.push_back(op); });
    for (TlaOpTy op : ops) {
        OpBuilder builder(op);
        Location loc = op.getLoc();
        Value lowered = isa<FloatType>(op.getResult().getType()) ?
                            builder.create<FloatOpTy>(loc, op.getLhs(), op.getRhs()).getResult() :
                            builder.create<IntOpTy>(loc, op.getLhs(), op.getRhs()).getResult();
        op.getResult().replaceAllUsesWith(lowered);
        op.erase();
    }
}

// The math ops are float-only, so they need no int/float dispatch -- one op in,
// one op out.
// The target has no bf16 transcendental unit -- exp, log, sqrt and pow
// all fail to select -- so those are computed in f32 and rounded back. For
// every other type the compute type is the source type and both casts below
// are omitted, so this costs nothing. The f32 intermediate is strictly more
// accurate than a native bf16 evaluation would be.
static Type simtMathComputeType(Type type)
{
    if (isa<BFloat16Type>(type))
        return Float32Type::get(type.getContext());
    return type;
}

static Value promoteForSimtMath(OpBuilder& builder, Location loc, Value value)
{
    Type computeType = simtMathComputeType(value.getType());
    if (computeType == value.getType())
        return value;
    // llvm.fpext, not arith.extf: convert-hivm-to-std rewrites the arith form
    // into a vcast_bfloat16_t_to_float_1d_with_mode vector-template call that the
    // c310 bitcode does not export.
    return builder.create<LLVM::FPExtOp>(loc, computeType, value);
}

static Value demoteAfterSimtMath(OpBuilder& builder, Location loc, Value value, Type resultType)
{
    if (value.getType() == resultType)
        return value;
    return builder.create<LLVM::FPTruncOp>(loc, resultType, value);
}

template <typename TlaOpTy, typename MathOpTy>
static void lowerSimtUnaryIn(func::FuncOp vf, bool promoteBF16 = true)
{
    SmallVector<TlaOpTy, 8> ops;
    vf.walk([&](TlaOpTy op) { ops.push_back(op); });
    for (TlaOpTy op : ops) {
        OpBuilder builder(op);
        Location loc = op.getLoc();
        Type resultType = op.getResult().getType();
        Value operand = promoteBF16 ? promoteForSimtMath(builder, loc, op.getOperand()) : op.getOperand();
        Value computed = builder.create<MathOpTy>(loc, operand).getResult();
        Value lowered = promoteBF16 ? demoteAfterSimtMath(builder, loc, computed, resultType) : computed;
        op.getResult().replaceAllUsesWith(lowered);
        op.erase();
    }
}

// Lower every tla.simt_* arithmetic op in the outlined vector function.
static void lowerSimtArithmeticIn(func::FuncOp vf)
{
    lowerSimtBinaryIn<::tla::SimtAddOp, arith::AddFOp, arith::AddIOp>(vf);
    lowerSimtBinaryIn<::tla::SimtSubOp, arith::SubFOp, arith::SubIOp>(vf);
    lowerSimtBinaryIn<::tla::SimtMulOp, arith::MulFOp, arith::MulIOp>(vf);
    lowerSimtBinaryIn<::tla::SimtDivOp, arith::DivFOp, arith::DivSIOp>(vf);
    // fmin/fmax semantics: the NaN-propagating forms have no instruction here.
    lowerSimtIntrinBinaryIn<::tla::SimtMaxOp, LLVM::MaxNumOp, LLVM::SMaxOp>(vf);
    lowerSimtIntrinBinaryIn<::tla::SimtMinOp, LLVM::MinNumOp, LLVM::SMinOp>(vf);
    lowerSimtCmpIn(vf);
    lowerSimtWhereIn(vf);
    lowerSimtCastIn(vf);

    SmallVector<::tla::SimtPowOp, 4> powOps;
    vf.walk([&](::tla::SimtPowOp op) { powOps.push_back(op); });
    for (::tla::SimtPowOp op : powOps) {
        OpBuilder builder(op);
        Location powLoc = op.getLoc();
        Type powResultType = op.getResult().getType();
        Value powLhs = promoteForSimtMath(builder, powLoc, op.getLhs());
        Value powRhs = promoteForSimtMath(builder, powLoc, op.getRhs());
        Value lowered = demoteAfterSimtMath(
            builder, powLoc, builder.create<math::PowFOp>(powLoc, powLhs, powRhs).getResult(), powResultType);
        op.getResult().replaceAllUsesWith(lowered);
        op.erase();
    }

    lowerSimtUnaryIn<::tla::SimtSqrtOp, math::SqrtOp>(vf);
    lowerSimtUnaryIn<::tla::SimtExpOp, math::ExpOp>(vf);
    lowerSimtUnaryIn<::tla::SimtAbsOp, math::AbsFOp>(vf, /*promoteBF16=*/false);

    // log goes straight to the LLVM intrinsic: math.log would be expanded by
    // convert-hivm-to-std into a vln_1d_float call that the c310 bitcode does
    // not export.
    SmallVector<::tla::SimtLogOp, 4> logOps;
    vf.walk([&](::tla::SimtLogOp op) { logOps.push_back(op); });
    for (::tla::SimtLogOp op : logOps) {
        OpBuilder builder(op);
        Location logLoc = op.getLoc();
        Type logResultType = op.getResult().getType();
        Value logOperand = promoteForSimtMath(builder, logLoc, op.getOperand());
        Value lowered = demoteAfterSimtMath(
            builder, logLoc, builder.create<LLVM::LogOp>(logLoc, logOperand.getType(), logOperand).getResult(),
            logResultType);
        op.getResult().replaceAllUsesWith(lowered);
        op.erase();
    }
}

// Raw device pointer for a memref, the three steps the SIMT ABI expects.
static Value memrefToRawPointer(OpBuilder& builder, Location loc, Value memref, int addressSpace)
{
    Value index = builder.create<mlir::memref::ExtractAlignedPointerAsIndexOp>(loc, memref);
    Value asI64 = builder.create<arith::IndexCastOp>(loc, builder.getI64Type(), index);
    auto ptrType = LLVM::LLVMPointerType::get(builder.getContext(), addressSpace);
    return builder.create<LLVM::IntToPtrOp>(loc, ptrType, asI64);
}

// tla.arch.sync_threads is a block-wide barrier between the threads of a SIMT
// vector function; it maps straight onto the regbase intrinsic.
static void lowerSimtSyncThreadsIn(func::FuncOp vf)
{
    SmallVector<::tla::SyncThreadsOp, 4> ops;
    vf.walk([&](::tla::SyncThreadsOp op) { ops.push_back(op); });
    for (::tla::SyncThreadsOp op : ops) {
        OpBuilder builder(op);
        builder.create<hivm_regbaseintrins::SyncThreadsOp>(op.getLoc());
        op.erase();
    }
}

// Element offset of a descriptor's view within its allocation, or a null Value
// when the view starts at the allocation's own base.
//
// The SIMT launch ABI passes a bare pointer -- there is no descriptor on the
// other side -- so a view that starts partway in has to have that start folded
// into the pointer. The formula is the one TlaLowerScalarAccessPass uses for the
// general path (coord[0]*stride[0] + coord[1]*stride[1], in elements); the two
// must agree or the same tensor means different things inside and outside a
// SIMT region.
static bool isZeroConstantValue(Value value)
{
    std::optional<int64_t> constant = getConstantIntValue(value);
    return constant && *constant == 0;
}

static FailureOr<Value> simtViewElementOffset(
    OpBuilder& builder, Location loc, const TensorDescriptor& desc, Operation* diagnosticOp)
{
    if (isZeroConstantValue(desc.coord[0]) && isZeroConstantValue(desc.coord[1]))
        return Value();

    // No contiguity requirement: linearizeSimtAccessesIn folds the tensor's real
    // strides into every index, so the pointer only has to carry the view's
    // *start*. A strided view -- tile_view of a wider parent at a non-zero
    // coordinate -- is handled by the two together: this offset moves the base,
    // and the linearized index walks the rows.

    Value rowOffset = builder.createOrFold<arith::MulIOp>(loc, desc.coord[0], desc.stride[0]);
    Value colOffset = builder.createOrFold<arith::MulIOp>(loc, desc.coord[1], desc.stride[1]);
    return builder.createOrFold<arith::AddIOp>(loc, rowOffset, colOffset);
}

// Collapse a rank-2 per-thread access into a single linear index, using the
// tensor's own strides.
//
// The SIMT launch passes one bare pointer per buffer and nothing else, so a
// memref parameter's strides are implicit and packed. A view whose rows are not
// packed -- tile_view of a wider parent, say (2,4) out of (2,8) where the rows
// are 8 apart -- cannot be described that way: memref<2x4xf32> addresses [1,2]
// as 1*4+2 instead of 1*8+2, silently reading the wrong row.
//
// Rewriting t[i, j] to flat[i*stride0 + j*stride1] removes the problem: the
// parameter stays flat and the real strides live in the arithmetic. Constant
// indices fold, so the common cases cost nothing.
static LogicalResult linearizeSimtAccessesIn(::tla::VecFuncOp vecFuncOp)
{
    // Logical dim d of a rank-R tensor lives at descriptor slot d + (2 - R): a
    // rank-1 tensor keeps its extent/stride in slot 1, a rank-2 one in slots 0
    // and 1. (The same convention TlaLowerScalarAccessPass uses.)
    auto descriptorSlot = [](size_t dim, size_t rank) { return dim + (2 - rank); };

    SmallVector<Operation*, 8> needsFolding;
    vecFuncOp.getBody().walk([&](Operation* op) {
        if (!isa<::tla::SimtLoadOp, ::tla::SimtStoreOp>(op))
            return;
        unsigned indexCount = op->getNumOperands() - (isa<::tla::SimtStoreOp>(op) ? 2 : 1);
        if (indexCount > 1) {
            needsFolding.push_back(op);
            return;
        }
        // A rank-1 view can be strided too -- every other element, say -- and its
        // single index has to be scaled or t[1] reads the neighbour. Note the
        // stride may be a *runtime* value, so this cannot be decided from the
        // static type alone: anything not provably 1 gets folded.
        if (indexCount == 1) {
            // Skip only what is *provably* unit-stride. When the descriptor is not
            // available, fall back to the static type; if that cannot prove stride 1
            // either, collect the op anyway so the loop below reports it. Silently
            // skipping here is what made a strided access read its neighbour.
            auto descOp = op->getOperand(0).getDefiningOp<::tla::TensorDescOp>();
            std::optional<int64_t> constant;
            if (descOp) {
                FailureOr<TensorDescriptor> desc = descriptorFromTensorDescOp(descOp);
                if (succeeded(desc))
                    constant = getConstantIntValue(desc->stride[1]);
            } else if (FailureOr<ParsedTensorInfo> info = parseTensorInfo(op->getOperand(0).getType());
                       succeeded(info) && !info->strides.empty() && !ShapedType::isDynamic(info->strides.back())) {
                constant = info->strides.back();
            }
            if (!constant || *constant != 1)
                needsFolding.push_back(op);
        }
    });

    // One narrowing per distinct stride value, not per access: each cast becomes a
    // captured scalar and therefore a launch argument, so emitting one per access
    // would grow the ABI linearly with the number of accesses.
    DenseMap<Value, Value> narrowedStrides;

    for (Operation* op : needsFolding) {
        bool isStore = isa<::tla::SimtStoreOp>(op);
        Value tensor = op->getOperand(0);

        // Take the strides from the descriptor rather than the type: they are SSA
        // values, so a stride computed at runtime (tla.arch.block_num() + 1, say)
        // works exactly like a constant one. The multiply lands inside the region,
        // which makes the stride a captured scalar -- the launch forwards it after
        // the buffers.
        auto descOp = tensor.getDefiningOp<::tla::TensorDescOp>();
        if (!descOp)
            return op->emitOpError(
                       "cannot linearize a per-thread access: expected a "
                       "materialized tla.tensor_desc"),
                   failure();
        FailureOr<TensorDescriptor> desc = descriptorFromTensorDescOp(descOp);
        if (failed(desc))
            return failure();

        // operands are: tensor, indices..., (value for a store)
        OperandRange indexRange = op->getOperands().drop_front(1);
        if (isStore)
            indexRange = indexRange.drop_back(1);
        SmallVector<Value, 2> indices(indexRange.begin(), indexRange.end());
        if (indices.size() != 1 && indices.size() != 2)
            return op->emitOpError("per-thread access expects one or two indices, got ") << indices.size(), failure();

        OpBuilder builder(op);
        Location loc = op->getLoc();
        Value linear;
        for (auto [dim, index] : llvm::enumerate(indices)) {
            Value stride = desc->stride[descriptorSlot(dim, indices.size())];
            Value scaled = index;
            std::optional<int64_t> constant = getConstantIntValue(stride);
            if (!constant || *constant != 1) {
                Value strideInRegion = stride;
                if (!constant) {
                    // A runtime stride crosses the launch as a captured scalar, and an
                    // index-typed launch argument reaches hivmc as an
                    // unrealized_conversion_cast and fails to translate. Narrow to i32
                    // outside the region -- that is what gets captured -- and widen back
                    // inside. NOTE: this truncates a stride of 2^31 or more; such a stride
                    // would exceed any addressable buffer on this target, but it is a
                    // silent bound rather than a checked one.
                    Value asI32 = narrowedStrides.lookup(stride);
                    if (!asI32) {
                        OpBuilder outside(vecFuncOp);
                        asI32 = outside.createOrFold<arith::IndexCastOp>(loc, outside.getI32Type(), stride);
                        narrowedStrides.insert({stride, asI32});
                    }
                    strideInRegion = builder.create<arith::IndexCastOp>(loc, builder.getIndexType(), asI32);
                }
                scaled = builder.createOrFold<arith::MulIOp>(loc, index, strideInRegion);
            }
            linear = linear ? builder.createOrFold<arith::AddIOp>(loc, linear, scaled) : scaled;
        }

        if (isStore) {
            builder.create<::tla::SimtStoreOp>(loc, tensor, ValueRange{linear}, op->getOperands().back());
        } else {
            Value loaded =
                builder.create<::tla::SimtLoadOp>(loc, op->getResult(0).getType(), tensor, ValueRange{linear});
            op->getResult(0).replaceAllUsesWith(loaded);
        }
        op->erase();
    }
    return success();
}

class LowerSimtVecFuncPattern : public OpRewritePattern<::tla::VecFuncOp> {
public:
    LowerSimtVecFuncPattern(
        MLIRContext* context, ModuleOp module, int& nextSimtRegionId, DenseMap<Value, Value>& baseMemrefCache)
        : OpRewritePattern<::tla::VecFuncOp>(context, /*benefit=*/3),
          module(module),
          nextSimtRegionId(nextSimtRegionId),
          baseMemrefCache(baseMemrefCache)
    {}

    LogicalResult matchAndRewrite(::tla::VecFuncOp vecFuncOp, PatternRewriter& rewriter) const override
    {
        if (!isSimtVecFunc(vecFuncOp))
            return rewriter.notifyMatchFailure(vecFuncOp, "not a SIMT tla.vec.func");
        if (vecFuncOp.getBody().empty())
            return rewriter.notifyMatchFailure(vecFuncOp, "expected tla.vec.func body");

        Location loc = vecFuncOp.getLoc();
        auto threadBlockDimAttr = vecFuncOp->getAttrOfType<DenseI64ArrayAttr>("thread_block_dim");
        if (!threadBlockDimAttr || threadBlockDimAttr.size() != 3)
            return vecFuncOp.emitOpError("SIMT tla.vec.func requires a 3-element 'thread_block_dim' attribute"),
                   failure();
        int64_t blockDimX = threadBlockDimAttr[0], blockDimY = threadBlockDimAttr[1], blockDimZ = threadBlockDimAttr[2];

        SimtCaptures captures;
        collectSimtCaptures(vecFuncOp, captures);
        // Runtime values computed outside the region (a core index, a block count,
        // a dynamic bound) are forwarded as extra launch arguments after the
        // buffers. Only scalars fit through that ABI.
        for (Value scalar : captures.scalars) {
            if (!scalar.getType().isIntOrIndexOrFloat())
                return vecFuncOp.emitOpError() << "a SIMT tla.vec.func can only capture scalar runtime values, "
                                                  "but this region reads one of type "
                                               << scalar.getType()
                                               << " computed outside it. Buffers must be captured as tensors; "
                                                  "anything else has no SIMT launch-argument form",
                       failure();
            // An index-typed launch argument survives TlaCompile but dies in hivmc
            // with "LLVM Translation failed for operation:
            // builtin.unrealized_conversion_cast". Integers and floats cross fine
            // (verified on device), so index is the one type that has to be narrowed
            // rather than rejected -- callers legitimately capture index-typed values
            // such as a tensor's stride.
            if (scalar.getType().isIndex())
                return vecFuncOp.emitOpError() << "a SIMT tla.vec.func cannot capture an index-typed runtime value "
                                                  "("
                                               << scalar.getType()
                                               << "): the launch ABI carries integers and floats, and an index "
                                                  "argument fails to translate in hivmc. Cast it to i32 outside the "
                                                  "region first",
                       failure();
        }

        // Base memref per captured tensor. These are the ABI buffers -- plain
        // static memrefs -- and become both the outlined parameters and, through
        // the pointer triple, the launch arguments.
        OpBuilder callBuilder(vecFuncOp);
        SmallVector<Value, 4> baseMemrefs;
        SmallVector<Type, 4> paramTypes;
        SmallVector<int, 4> addressSpaces;
        SmallVector<Value, 4> viewOffsets; // null == view starts at the base
        SmallVector<Type, 4> elementTypes;
        for (auto [operandIndex, tensor] : llvm::enumerate(captures.tensors)) {
            auto descOp = tensor.getDefiningOp<::tla::TensorDescOp>();
            if (!descOp)
                return vecFuncOp.emitOpError("SIMT capture expects a materialized tla.tensor_desc"), failure();
            FailureOr<TensorDescriptor> desc = descriptorFromTensorDescOp(descOp);
            if (failed(desc))
                return failure();

            // Address space and static-shape legality were already reported by
            // checkSimtRegionCaptures before the driver ran; bail quietly here.
            std::optional<int> addressSpace = simtAddressSpaceFor(desc->addrspace);
            if (!addressSpace)
                return failure();
            addressSpaces.push_back(*addressSpace);

            FailureOr<Value> base = getOrMaterializeDescriptorBaseMemref(
                callBuilder, loc, *desc, vecFuncOp.getOperation(), baseMemrefCache);
            if (failed(base))
                return failure();
            auto baseType = dyn_cast<MemRefType>((*base).getType());
            if (!baseType)
                return vecFuncOp.emitOpError("SIMT capture did not materialize as a memref; got ") << (*base).getType(),
                       failure();
            baseMemrefs.push_back(*base);
            elementTypes.push_back(baseType.getElementType());

            // Fold the view's start into the pointer we hand the launch, so two views
            // of one allocation stay distinguishable on the other side.
            FailureOr<Value> viewOffset = simtViewElementOffset(callBuilder, loc, *desc, vecFuncOp.getOperation());
            if (failed(viewOffset))
                return failure();
            viewOffsets.push_back(*viewOffset);
            // Parameters are always flat. One bare pointer per buffer crosses the
            // launch ABI, so a memref's implicit packed strides are the only ones it
            // could ever express; rank-2 accesses were linearized above with the
            // tensor's real strides, so the parameter only has to *span* the view --
            // the largest index that arithmetic can produce, plus one.
            FailureOr<ParsedTensorInfo> info = parseTensorInfo(tensor.getType());
            if (failed(info))
                return failure();
            size_t logicalRank = info->shape.size();
            if (logicalRank != 1 && logicalRank != 2)
                return vecFuncOp.emitOpError()
                           << "a SIMT tla.vec.func captures tensors of logical rank 1 or 2, got " << logicalRank,
                       failure();

            int64_t extent = ShapedType::kDynamic;
            if (logicalRank == 2 && info->strides.size() == 2 && !ShapedType::isDynamic(info->shape[0]) &&
                !ShapedType::isDynamic(info->shape[1]) && !ShapedType::isDynamic(info->strides[0]) &&
                !ShapedType::isDynamic(info->strides[1])) {
                // A strided view reaches past rows*cols: its last row starts at
                // (rows-1)*stride0, so bounding by rows*cols would cut the buffer short
                // and the verifier would reject the very access we just linearized.
                extent = (info->shape[0] - 1) * info->strides[0] + (info->shape[1] - 1) * info->strides[1] + 1;
            } else if (*viewOffset) {
                // The pointer now starts at the view, so the parameter must be bounded
                // by the view too: keeping the allocation's extent would let an index
                // run off the end of the buffer.
                if (std::optional<int64_t> constant = getConstantIntValue(desc->shape[1]))
                    extent = *constant;
            } else if (baseType.getRank() == 1 && !baseType.isDynamicDim(0)) {
                extent = baseType.getDimSize(0);
            } else if (std::optional<int64_t> constant = getConstantIntValue(desc->shape[1])) {
                extent = *constant;
            }
            if (ShapedType::isDynamic(extent))
                return failure();
            SmallVector<int64_t, 1> paramShape{extent};
            (void)operandIndex;
            // The parameter must carry an *integer* address space, not
            // #hivm.address_space<...>: only then does memref-to-llvm lower it to a
            // bare pointer matching the !llvm.ptr the launch passes. With the hivm
            // attribute it becomes a descriptor struct and hivmc-a5 aborts with
            // "Calling a function with a bad signature!".
            paramTypes.push_back(MemRefType::get(
                paramShape, baseType.getElementType(), MemRefLayoutAttrInterface{},
                rewriter.getI64IntegerAttr(*addressSpace)));
        }

        // ---- the outlined vector function ----
        auto parentFunc = vecFuncOp->getParentOfType<func::FuncOp>();
        std::string vfName = (parentFunc ? parentFunc.getName().str() : std::string("kernel")) + "_vf_simt" +
                             (nextSimtRegionId ? std::to_string(nextSimtRegionId) : std::string());
        ++nextSimtRegionId;

        for (Value scalar : captures.scalars)
            paramTypes.push_back(scalar.getType());

        ModuleOp mod = module;
        OpBuilder moduleBuilder(mod.getBodyRegion());
        moduleBuilder.setInsertionPointToEnd(mod.getBody());
        auto vfType = FunctionType::get(rewriter.getContext(), paramTypes, {});
        auto vf = moduleBuilder.create<func::FuncOp>(loc, vfName, vfType);
        MLIRContext* ctx = rewriter.getContext();
        vf->setAttr(
            hivm_regbaseintrins::kDavinciCallingConvAttrName,
            hivm_regbaseintrins::SIMT_EntryAttr::get(ctx, static_cast<uint32_t>(blockDimX * blockDimY * blockDimZ)));
        vf->setAttr("noinline", BoolAttr::get(ctx, false));
        // SIMT regions are always vector-core functions. A pure AIV module gets
        // away without this attribute because hivm.module_core_type covers the
        // whole module, but a mixed AIC+AIV module has no single core type, and
        // hivmc-a5 rejects the outlined function with "Unknown core type". SIMT
        // is inherently vector-core work, so mark it AIV outright rather than
        // deriving it from the enclosing function or the pass order.
        vf->setAttr(hivm::TFuncCoreTypeAttr::name, hivm::TFuncCoreTypeAttr::get(ctx, hivm::TFuncCoreType::AIV));

        Block* vfBody = vf.addEntryBlock();
        OpBuilder bodyBuilder(vfBody, vfBody->begin());
        IRMapping mapping;
        ArrayRef<BlockArgument> vfArgs = vfBody->getArguments();
        for (auto [tensor, arg] : llvm::zip_equal(captures.tensors, vfArgs.take_front(captures.tensors.size())))
            mapping.map(tensor, arg);
        for (auto [scalar, arg] : llvm::zip_equal(captures.scalars, vfArgs.drop_front(captures.tensors.size())))
            mapping.map(scalar, arg);
        for (Operation* constant : captures.constants)
            bodyBuilder.clone(*constant, mapping);

        // Clone the region body in. Tensor operands are remapped to the memref
        // parameters as we go, which leaves the cloned scalar accesses momentarily
        // holding a memref where a !tla.tensor is declared; the walk below converts
        // them to memref.load/store, at whatever nesting depth they sit.
        for (Operation& op : vecFuncOp.getBody().front())
            bodyBuilder.clone(op, mapping);
        bodyBuilder.create<func::ReturnOp>(loc);

        if (failed(rewriteSimtAccessesOntoParams(vf)))
            return failure();
        lowerSimtTripleOpsIn<
            ::tla::ThreadIdxOp, hivm_regbaseintrins::ThreadIdXOp, hivm_regbaseintrins::ThreadIdYOp,
            hivm_regbaseintrins::ThreadIdZOp>(vf);
        lowerSimtTripleOpsIn<
            ::tla::ThreadBlockDimOp, hivm_regbaseintrins::BlockDimXOp, hivm_regbaseintrins::BlockDimYOp,
            hivm_regbaseintrins::BlockDimZOp>(vf);
        lowerSimtArithmeticIn(vf);
        lowerSimtSyncThreadsIn(vf);

        // ---- the launch ----
        SmallVector<Value, 4> launchArgs;
        for (auto [base, addressSpace, viewOffset, elementType] :
             llvm::zip_equal(baseMemrefs, addressSpaces, viewOffsets, elementTypes)) {
            Value pointer = memrefToRawPointer(callBuilder, loc, base, addressSpace);
            if (viewOffset) {
                Value index = callBuilder.createOrFold<arith::IndexCastOp>(loc, callBuilder.getI64Type(), viewOffset);
                auto pointerType = LLVM::LLVMPointerType::get(callBuilder.getContext(), addressSpace);
                pointer = callBuilder.create<LLVM::GEPOp>(loc, pointerType, elementType, pointer, ValueRange{index});
            }
            launchArgs.push_back(pointer);
        }
        // Captured runtime scalars follow the buffers, in capture order: the ABI is
        // positional, so this must match the parameter list built above.
        for (Value scalar : captures.scalars)
            launchArgs.push_back(scalar);
        auto i64 = callBuilder.getI64Type();
        Value tx = callBuilder.create<arith::ConstantOp>(loc, i64, callBuilder.getI64IntegerAttr(blockDimX));
        Value ty = callBuilder.create<arith::ConstantOp>(loc, i64, callBuilder.getI64IntegerAttr(blockDimY));
        Value tz = callBuilder.create<arith::ConstantOp>(loc, i64, callBuilder.getI64IntegerAttr(blockDimZ));
        callBuilder.create<hivm_regbaseintrins::LaunchFuncOp>(
            loc, FlatSymbolRefAttr::get(ctx, vfName), tx, ty, tz, launchArgs);

        rewriter.eraseOp(vecFuncOp);
        return success();
    }

private:
    // Convert every cloned tla.simt_load/simt_store onto the memref parameter its
    // tensor operand was mapped to.
    static LogicalResult rewriteSimtAccessesOntoParams(func::FuncOp vf)
    {
        SmallVector<Operation*, 8> stale;
        vf.walk([&](Operation* op) {
            if (isa<::tla::SimtLoadOp, ::tla::SimtStoreOp>(op))
                stale.push_back(op);
        });
        for (Operation* op : stale) {
            OpBuilder builder(op);
            // The clone remapped the tensor operand to a memref parameter, so the op
            // no longer matches its own declared operand type. Read that operand
            // positionally: the generated getSource()/getDest() accessors cast to
            // TypedValue<TlaTensorType>, which asserts in an assertions-enabled build
            // (and silently returned a mistyped value in one without).
            if (auto load = dyn_cast<::tla::SimtLoadOp>(op)) {
                Value source = op->getOperand(0);
                SmallVector<Value, 2> indices(load.getIndices().begin(), load.getIndices().end());
                if (!isa<MemRefType>(source.getType()))
                    return load.emitOpError("tla.simt_load source did not map to a memref parameter"), failure();
                Value loaded = builder.create<mlir::memref::LoadOp>(load.getLoc(), source, indices);
                load.getResult().replaceAllUsesWith(loaded);
            } else {
                auto store = cast<::tla::SimtStoreOp>(op);
                Value dest = op->getOperand(0);
                SmallVector<Value, 2> indices(store.getIndices().begin(), store.getIndices().end());
                if (!isa<MemRefType>(dest.getType()))
                    return store.emitOpError("tla.simt_store dest did not map to a memref parameter"), failure();
                builder.create<mlir::memref::StoreOp>(store.getLoc(), store.getValue(), dest, indices);
            }
            op->erase();
        }
        return success();
    }

    ModuleOp module;
    int& nextSimtRegionId;
    DenseMap<Value, Value>& baseMemrefCache;
};

class LowerVecFuncRegionPattern : public OpRewritePattern<::tla::VecFuncOp> {
public:
    LowerVecFuncRegionPattern(
        MLIRContext* context, ModuleOp module, int& nextVectorRegionId, DenseMap<Value, Value>& loweredMemrefByValue,
        bool& invalidScalarAccessBase)
        : OpRewritePattern<::tla::VecFuncOp>(context, /*benefit=*/2),
          module(module),
          nextVectorRegionId(nextVectorRegionId),
          loweredMemrefByValue(loweredMemrefByValue),
          invalidScalarAccessBase(invalidScalarAccessBase)
    {}

    LogicalResult matchAndRewrite(::tla::VecFuncOp vecFuncOp, PatternRewriter& rewriter) const override
    {
        // SIMD is the only mode this outlining path models: it maps the region onto
        // whole-vector AVE instructions. SIMT regions describe per-thread scalar
        // work and are lowered elsewhere.
        if (!isSimdVecFunc(vecFuncOp))
            return rewriter.notifyMatchFailure(vecFuncOp, "not a SIMD tla.vec.func");

        auto* body = vecFuncOp.getBody().empty() ? nullptr : &vecFuncOp.getBody().front();
        if (!body)
            return rewriter.notifyMatchFailure(vecFuncOp, "expected tla.vec.func body");

        // Collect the load / binary compute / store ops (used for arg dedup and
        // graph validation); the helper builder walks the region itself to carry
        // the control flow structure.
        SmallVector<::tla::LoadOp, 4> loads;
        SmallVector<::tla::FullOp, 4> fulls;
        SmallVector<::tla::CreateMaskOp, 4> createMasks;
        SmallVector<::tla::UpdateMaskOp, 4> updateMasks;
        SmallVector<::tla::ArangeOp, 4> aranges;
        SmallVector<Operation*, 4> computeOps;
        SmallVector<::tla::StoreOp, 2> stores;
        vecFuncOp->walk([&](Operation* op) {
            if (auto load = dyn_cast<::tla::LoadOp>(op)) {
                loads.push_back(load);
            } else if (auto full = dyn_cast<::tla::FullOp>(op)) {
                fulls.push_back(full);
            } else if (auto createMask = dyn_cast<::tla::CreateMaskOp>(op)) {
                createMasks.push_back(createMask);
            } else if (auto updateMask = dyn_cast<::tla::UpdateMaskOp>(op)) {
                updateMasks.push_back(updateMask);
            } else if (auto arange = dyn_cast<::tla::ArangeOp>(op)) {
                aranges.push_back(arange);
            } else if (auto store = dyn_cast<::tla::StoreOp>(op)) {
                stores.push_back(store);
            } else if (isVectorComputeOp(op)) {
                computeOps.push_back(op);
            }
            return WalkResult::advance();
        });
        if (stores.empty()) {
            // Scalar-only (or empty) VF cannot be outlined as a BiSheng helper — that
            // path requires tla.store. If there is also no tile load/compute, inline
            // the body into the parent (same as tla.vector flattening) so
            // scalar_load/store + scf stay legal for later convert-scf-to-cf.
            if (!loads.empty() || !fulls.empty() || !createMasks.empty() || !updateMasks.empty() || !aranges.empty() ||
                !computeOps.empty())
                return rewriter.notifyMatchFailure(vecFuncOp, "expected tla.vec.func body with a tla.store");
            rewriter.inlineBlockBefore(body, vecFuncOp->getBlock(), vecFuncOp->getIterator());
            rewriter.eraseOp(vecFuncOp);
            return success();
        }

        // Validate the graph: every compute operand and store source must come from
        // a tla.load result or a prior compute result inside this region.
        DenseSet<Value> producedValues;
        for (::tla::LoadOp load : loads) {
            producedValues.insert(load.getResult());
            if (Value result2 = load.getResult2())
                producedValues.insert(result2);
        }
        for (::tla::FullOp full : fulls)
            producedValues.insert(full.getResult());
        for (::tla::CreateMaskOp createMask : createMasks)
            producedValues.insert(createMask.getResult());
        for (::tla::UpdateMaskOp updateMask : updateMasks)
            producedValues.insert(updateMask.getMask());
        for (::tla::ArangeOp arange : aranges)
            producedValues.insert(arange.getResult());
        auto isRegisterCarrier = [](Value value) {
            Type type = value.getType();
            return isa<::tla::VectorSSAType, ::tla::MaskSSAType>(type);
        };
        vecFuncOp.walk([&](scf::ForOp forOp) {
            for (BlockArgument arg : forOp.getRegionIterArgs())
                if (isRegisterCarrier(arg))
                    producedValues.insert(arg);
            for (Value result : forOp.getResults())
                if (isRegisterCarrier(result))
                    producedValues.insert(result);
        });
        vecFuncOp.walk([&](scf::IfOp ifOp) {
            for (Value result : ifOp.getResults())
                if (isRegisterCarrier(result))
                    producedValues.insert(result);
        });

        for (Operation* computeOp : computeOps) {
            if (isa<::tla::InterleaveOp>(computeOp) || isa<::tla::DeInterleaveOp>(computeOp)) {
                if (computeOp->getNumResults() != 2)
                    return rewriter.notifyMatchFailure(vecFuncOp, "unexpected two-result tla compute op shape");
            } else if (computeOp->getNumResults() != 1) {
                return rewriter.notifyMatchFailure(vecFuncOp, "unexpected tla compute op shape");
            }
            if (auto anyInfo = getAnyVectorOperationInfo(computeOp)) {
                if (auto info = anyInfo->binary) {
                    // Vector operands must come from a load or prior compute op. A
                    // vector-scalar rhs may be captured or cloned into the helper.
                    TlaBinaryOperands ops = info->operands;
                    if (!ops.lhs || !ops.rhs || !producedValues.contains(ops.lhs))
                        return rewriter.notifyMatchFailure(
                            vecFuncOp,
                            "expected binary op operand from load/create/update mask "
                            "or prior compute op");
                    if (info->rhsKind == VectorRhsKind::Vector && !producedValues.contains(ops.rhs))
                        return rewriter.notifyMatchFailure(
                            vecFuncOp,
                            "expected binary op rhs from load/create/update mask "
                            "or prior compute op");
                    if (ops.mask && !producedValues.contains(ops.mask))
                        return rewriter.notifyMatchFailure(
                            vecFuncOp,
                            "expected binary op mask from create/update mask, compare, "
                            "or prior mask compute op");
                } else if (auto unaryInfo = anyInfo->unary) {
                    TlaUnaryOperands ops = unaryInfo->operands;
                    if (!ops.operand || !producedValues.contains(ops.operand))
                        return rewriter.notifyMatchFailure(
                            vecFuncOp,
                            "expected unary op operand from load/create/update mask "
                            "or prior compute op");
                    if (ops.mask && !producedValues.contains(ops.mask))
                        return rewriter.notifyMatchFailure(
                            vecFuncOp,
                            "expected unary op mask from create/update mask, compare, "
                            "or prior mask compute op");
                } else {
                    return rewriter.notifyMatchFailure(vecFuncOp, "unexpected tla compute op");
                }
            } else if (auto cmpOp = dyn_cast<::tla::CmpOp>(computeOp)) {
                if (!producedValues.contains(cmpOp.getLhs()))
                    return rewriter.notifyMatchFailure(
                        vecFuncOp, "expected tla.cmp lhs from tla.load or prior compute op");
                if (isa<::tla::VectorSSAType>(cmpOp.getRhs().getType()) && !producedValues.contains(cmpOp.getRhs()))
                    return rewriter.notifyMatchFailure(
                        vecFuncOp, "expected tla.cmp rhs from tla.load or prior compute op");
                if (cmpOp.getMask() && !producedValues.contains(cmpOp.getMask()))
                    return rewriter.notifyMatchFailure(
                        vecFuncOp,
                        "expected tla.cmp mask from create/update mask or "
                        "prior mask compute op");
            } else if (auto whereOp = dyn_cast<::tla::WhereOp>(computeOp)) {
                if (!producedValues.contains(whereOp.getMask()))
                    return rewriter.notifyMatchFailure(
                        vecFuncOp,
                        "expected tla.where mask from create/update mask, compare, "
                        "SCF carrier, or prior mask compute op");
                if (!producedValues.contains(whereOp.getX()) || !producedValues.contains(whereOp.getY()))
                    return rewriter.notifyMatchFailure(
                        vecFuncOp, "expected tla.where operand from tla.load or prior compute op");
            } else if (auto squeezeOp = dyn_cast<::tla::SqueezeOp>(computeOp)) {
                if (!producedValues.contains(squeezeOp.getSrc()))
                    return rewriter.notifyMatchFailure(
                        vecFuncOp, "expected tla.squeeze src from tla.load or prior compute op");
                if (!producedValues.contains(squeezeOp.getMask()))
                    return rewriter.notifyMatchFailure(
                        vecFuncOp,
                        "expected tla.squeeze mask from create/update mask or "
                        "prior mask compute op");
            } else if (auto reduceOp = dyn_cast<::tla::ReduceOp>(computeOp)) {
                Value operand = reduceOp.getOperand();
                if (reduceOp.getMask() && !producedValues.contains(reduceOp.getMask()))
                    return rewriter.notifyMatchFailure(
                        vecFuncOp, "expected tla.reduce mask from a legal mask producer");
                if (!producedValues.contains(operand))
                    return rewriter.notifyMatchFailure(
                        vecFuncOp, "expected tla.reduce operand from tla.load or prior compute op");
            } else if (auto interleaveOp = dyn_cast<::tla::InterleaveOp>(computeOp)) {
                if (!producedValues.contains(interleaveOp.getSrc0()) ||
                    !producedValues.contains(interleaveOp.getSrc1()))
                    return rewriter.notifyMatchFailure(
                        vecFuncOp, "expected tla.interleave operands from tla.load or prior compute op");
            } else if (auto deInterleaveOp = dyn_cast<::tla::DeInterleaveOp>(computeOp)) {
                if (!producedValues.contains(deInterleaveOp.getSrc0()) ||
                    !producedValues.contains(deInterleaveOp.getSrc1()))
                    return rewriter.notifyMatchFailure(
                        vecFuncOp, "expected tla.deinterleave operands from tla.load or prior compute op");
            } else if (auto gatherOp = dyn_cast<::tla::GatherOp>(computeOp)) {
                if (gatherOp.getMask() && !producedValues.contains(gatherOp.getMask()))
                    return rewriter.notifyMatchFailure(
                        vecFuncOp, "expected tla.gather mask from a legal mask producer");
                if (!producedValues.contains(gatherOp.getY()))
                    return rewriter.notifyMatchFailure(
                        vecFuncOp, "expected tla.gather y operand from tla.load or prior compute op");
            } else if (auto castOp = dyn_cast<::tla::CastOp>(computeOp)) {
                if (castOp.getMask() && !producedValues.contains(castOp.getMask()))
                    return rewriter.notifyMatchFailure(vecFuncOp, "expected tla.cast mask from a legal mask producer");
                if (!producedValues.contains(castOp.getSource()))
                    return rewriter.notifyMatchFailure(
                        vecFuncOp, "expected tla.cast source from tla.load or prior compute op");
            } else {
                return rewriter.notifyMatchFailure(vecFuncOp, "unexpected tla compute op");
            }
            for (Value result : computeOp->getResults())
                producedValues.insert(result);
        }
        for (::tla::StoreOp store : stores) {
            if (store.getMask() && !producedValues.contains(store.getMask()))
                return rewriter.notifyMatchFailure(vecFuncOp, "expected tla.store mask from a legal mask producer");
            if (!producedValues.contains(store.getSource()))
                return rewriter.notifyMatchFailure(vecFuncOp, "expected tla.store source from tla.load or compute op");
        }

        auto funcOp = vecFuncOp->getParentOfType<func::FuncOp>();
        if (!funcOp)
            return rewriter.notifyMatchFailure(vecFuncOp, "expected enclosing func.func");

        // The helper takes one flat on-chip memref per referenced tensor, in body
        // order. Its extent is static when known; address-backed dynamic cases use
        // a rank-1 dynamic memref whose descriptor size is an explicit zero
        // sentinel. The helper must only consume that value as an address-backed
        // source for fixed-size reinterpret_cast tiles, never as capacity metadata.
        // Compute the operand list once and use it for both the helper signature
        // and the call.
        SmallVector<Value> helperOperands;
        DenseMap<Value, Value> helperOperandAliases;
        DominanceInfo dominanceInfo(funcOp);
        if (failed(collectVectorHelperOperands(
                body, vecFuncOp.getBody(), vecFuncOp, dominanceInfo, helperOperands, helperOperandAliases))) {
            if (!invalidScalarAccessBase)
                vecFuncOp.emitOpError(
                    "cannot outline scalar access because its base memref does not "
                    "dominate the vector helper call site; materialize dynamic "
                    "pointer-backed storage outside tla.vec.func");
            invalidScalarAccessBase = true;
            return failure();
        }
        if (helperOperands.empty())
            return rewriter.notifyMatchFailure(vecFuncOp, "expected vector region tensor operands");
        // Scalars captured from outside the region (e.g. a sub_block_idx computed at
        // the top of the kernel) are passed as trailing scalar arguments.
        SmallVector<Value> scalarOperands;
        collectVectorHelperScalarOperands(vecFuncOp, scalarOperands);

        auto helperOr = buildHelperFunc(
            module, funcOp, vecFuncOp, helperOperands, helperOperandAliases, scalarOperands, nextVectorRegionId,
            loweredMemrefByValue);
        if (failed(helperOr))
            return rewriter.notifyMatchFailure(vecFuncOp, "failed to build vector helper function");
        auto helper = *helperOr;

        // The for/if control flow lives inside the helper, so this is a single
        // call passing the helper memrefs that replaces the whole vec.func region.
        rewriter.setInsertionPoint(vecFuncOp);
        SmallVector<Value, 8> callOperands;
        callOperands.reserve(helperOperands.size());
        Value unknownExtent;
        for (Value tensor : helperOperands) {
            // Bridged scalar-access memref operands are passed as-is.
            if (isa<MemRefType>(tensor.getType())) {
                callOperands.push_back(tensor);
                continue;
            }
            auto type = getVectorHelperArgMemrefType(tensor);
            if (failed(type))
                return rewriter.notifyMatchFailure(vecFuncOp, "failed to type on-chip memref for vector helper call");
            // Materialize address-backed tla.tensor_desc operands at the call site.
            // tla-lower-tensor-desc is the sole descriptor producer, so every helper
            // operand here is a tensor_desc (raw memrefs were passed through above);
            // materialize its inttoptr base as a rank-1 helper arg when ptr-backed.
            Value ptr;
            if (auto descOp = tensor.getDefiningOp<::tla::TensorDescOp>()) {
                if (llvm::isa<::tla::PtrType>(descOp.getBase().getType()))
                    ptr = descOp.getBase();
            }
            FailureOr<Value> base = failure();
            if (ptr) {
                if (!ptr.getDefiningOp<::tla::IntToPtrOp>())
                    return rewriter.notifyMatchFailure(vecFuncOp, "expected pointer lowered to tla.inttoptr boundary");
                SmallVector<Value, 1> dynamicSizes;
                if (!type->hasStaticShape()) {
                    if (type->getRank() != 1 || type->getDimSize(0) != ShapedType::kDynamic)
                        return rewriter.notifyMatchFailure(
                            vecFuncOp, "expected flattened rank-1 dynamic vector helper memref");
                    if (!unknownExtent)
                        unknownExtent = rewriter.create<arith::ConstantIndexOp>(vecFuncOp.getLoc(), 0);
                    dynamicSizes.push_back(unknownExtent);
                }
                base = materializePtrValueAsMemref(
                    rewriter, vecFuncOp.getLoc(), ptr, *type, vecFuncOp.getOperation(), dynamicSizes);
                if (failed(base))
                    return rewriter.notifyMatchFailure(
                        vecFuncOp, "failed to materialize address-backed vector helper operand");
            } else {
                base = materializeBaseMemref(
                    rewriter, vecFuncOp.getLoc(), tensor,
                    /*loweredMemrefByValue=*/nullptr);
                if (failed(base))
                    return rewriter.notifyMatchFailure(vecFuncOp, "failed to materialize vector helper memref");
            }
            auto arg = castMemrefToExpected(rewriter, vecFuncOp.getLoc(), *base, *type);
            if (failed(arg))
                return rewriter.notifyMatchFailure(vecFuncOp, "failed to cast helper operand to expected memref type");
            callOperands.push_back(*arg);
        }
        // Captured scalars are defined in the parent (before this region), so they
        // dominate the call — pass them directly as trailing call operands.
        for (Value scalar : scalarOperands)
            callOperands.push_back(scalar);

        auto call = rewriter.create<func::CallOp>(vecFuncOp.getLoc(), helper, callOperands);
        call->setAttr("hivm.vector_function", UnitAttr::get(rewriter.getContext()));
        call->setAttr("no_inline", UnitAttr::get(rewriter.getContext()));
        rewriter.eraseOp(vecFuncOp);
        return success();
    }

private:
    ModuleOp module;
    int& nextVectorRegionId;
    DenseMap<Value, Value>& loweredMemrefByValue;
    bool& invalidScalarAccessBase;
};

class LowerCopyPattern : public OpRewritePattern<::tla::CopyOp> {
public:
    LowerCopyPattern(MLIRContext* context, DenseMap<Value, Value>& loweredMemrefByValue)
        : OpRewritePattern<::tla::CopyOp>(context, /*benefit=*/3), loweredMemrefByValue(loweredMemrefByValue)
    {}

    LogicalResult matchAndRewrite(::tla::CopyOp copyOp, PatternRewriter& rewriter) const override
    {
        if (copyOp->getNumOperands() != 2 || copyOp->getNumResults() != 0)
            return rewriter.notifyMatchFailure(copyOp, "expected tla.copy with 2 operands and 0 results");

        Value dstTile = copyOp.getDst();
        Value srcTile = copyOp.getSrc();
        auto dstDescOp = dstTile.getDefiningOp<::tla::TensorDescOp>();
        auto srcDescOp = srcTile.getDefiningOp<::tla::TensorDescOp>();
        if (!dstDescOp || !srcDescOp)
            return rewriter.notifyMatchFailure(
                copyOp, "expected tla.tensor_desc operand materialized by tla-lower-tensor-desc");
        auto dstDescOr = ::tla::descriptorFromTensorDescOp(dstDescOp);
        auto srcDescOr = ::tla::descriptorFromTensorDescOp(srcDescOp);
        const TensorDescriptor& dstDesc = *dstDescOr;
        const TensorDescriptor& srcDesc = *srcDescOr;

        std::string calleeName = ::tla::getCopyRouteCallee(
            copyOp.getContext(), srcDesc.addrspace, dstDesc.addrspace, srcDesc.layoutTag, dstDesc.layoutTag,
            srcDesc.elementType, dstDesc.elementType);
        if (calleeName.empty())
            return rewriter.notifyMatchFailure(
                copyOp, "unsupported tla.copy route (vector pass handles gm<->ub and ub->l1)");

        auto buildRuntimeMemref = [&](const TensorDescriptor& desc) -> FailureOr<Value> {
            FailureOr<Value> baseMemref = ::tla::getOrMaterializeDescriptorBaseMemref(
                rewriter, copyOp.getLoc(), desc, copyOp.getOperation(), loweredMemrefByValue);
            if (failed(baseMemref))
                return failure();
            auto baseType = dyn_cast<MemRefType>((*baseMemref).getType());
            if (!baseType)
                return failure();
            MemRefType runtimeType = ::tla::getDynamicStridedMemrefType(baseType);
            return ::tla::castMemrefToType(rewriter, copyOp.getLoc(), *baseMemref, runtimeType);
        };
        FailureOr<Value> srcRuntimeMemref = buildRuntimeMemref(srcDesc);
        FailureOr<Value> dstRuntimeMemref = buildRuntimeMemref(dstDesc);
        if (failed(srcRuntimeMemref) || failed(dstRuntimeMemref))
            return failure();
        SmallVector<Value, 24> payload = ::tla::buildCopyPayloadForRoute(rewriter, copyOp.getLoc(), srcDesc, dstDesc);
        SmallVector<Type, 22> operandTypes = {(*srcRuntimeMemref).getType(), (*dstRuntimeMemref).getType()};
        operandTypes.reserve(2 + payload.size());
        for (Value v : payload)
            operandTypes.push_back(v.getType());
        SmallVector<Value, 22> operands = {*srcRuntimeMemref, *dstRuntimeMemref};
        operands.append(payload.begin(), payload.end());
        auto callee = ::tla::getOrCreateRuntimeCall(copyOp->getParentOfType<ModuleOp>(), calleeName, operandTypes);

        // Enclose `copy` with atomic add and atomic none
        auto getAtomicKind = [](AtomicMode mode) -> hivm::AtomicKind {
            switch (mode) {
                case AtomicMode::add:
                    return hivm::AtomicKind::ADD;
                // For further extension, add other atomic mode case
                default:
                    return hivm::AtomicKind::NONE;
            }
        };

        auto atomicModeAttr = copyOp->getAttrOfType<::tla::AtomicModeAttr>("atomic_mode");
        Type dstType = cast<MemRefType>((*dstRuntimeMemref).getType()).getElementType();
        bool _enable_atomic = atomicModeAttr && atomicModeAttr.getAtomicMode() != AtomicMode::none;
        if (_enable_atomic) {
            if (atomicModeAttr.getAtomicMode() != AtomicMode::add) {
                copyOp.emitError() << "currently only atomic add is supported";
                return failure();
            }

            auto modeAttr =
                hivm::AtomicKindAttr::get(rewriter.getContext(), getAtomicKind(atomicModeAttr.getAtomicMode()));
            rewriter.create<hivm::SetAtomicOp>(copyOp.getLoc(), modeAttr, mlir::TypeAttr::get(dstType));
        }
        rewriter.create<func::CallOp>(copyOp.getLoc(), callee, operands);
        if (_enable_atomic) {
            auto modeAttr = hivm::AtomicKindAttr::get(rewriter.getContext(), hivm::AtomicKind::NONE);
            rewriter.create<hivm::SetAtomicOp>(copyOp.getLoc(), modeAttr, mlir::TypeAttr::get(dstType));
        }
        rewriter.eraseOp(copyOp);
        return success();
    }

private:
    DenseMap<Value, Value>& loweredMemrefByValue;
};

class InlineVectorRegionWrapperPattern : public OpRewritePattern<::tla::VectorOp> {
public:
    explicit InlineVectorRegionWrapperPattern(MLIRContext* context)
        : OpRewritePattern<::tla::VectorOp>(context, /*benefit=*/10)
    {}

    LogicalResult matchAndRewrite(::tla::VectorOp vectorOp, PatternRewriter& rewriter) const override
    {
        if (vectorOp->getNumRegions() == 0 || vectorOp.getBody().empty()) {
            rewriter.eraseOp(vectorOp);
            return success();
        }
        Block* body = &vectorOp.getBody().front();
        rewriter.inlineBlockBefore(body, vectorOp->getBlock(), vectorOp->getIterator());
        rewriter.eraseOp(vectorOp);
        return success();
    }
};

static void inlineVectorRegionWrappers(func::FuncOp funcOp)
{
    SmallVector<::tla::VectorOp, 4> wrappers;
    funcOp.walk([&](::tla::VectorOp vectorOp) { wrappers.push_back(vectorOp); });

    IRRewriter rewriter(funcOp.getContext());
    for (::tla::VectorOp vectorOp : wrappers) {
        if (!vectorOp)
            continue;
        if (vectorOp->getNumRegions() == 0 || vectorOp.getBody().empty()) {
            rewriter.eraseOp(vectorOp);
            continue;
        }
        Block* body = &vectorOp.getBody().front();
        rewriter.inlineBlockBefore(body, vectorOp->getBlock(), vectorOp->getIterator());
        rewriter.eraseOp(vectorOp);
    }
}

static void populateTlaToVectorPatterns(
    RewritePatternSet& patterns, ModuleOp module, int& nextVectorRegionId, int& nextSimtRegionId,
    DenseMap<Value, Value>& loweredMemrefByValue, bool& invalidScalarAccessBase)
{
    MLIRContext* ctx = patterns.getContext();
    patterns.add<InlineVectorRegionWrapperPattern>(ctx);
    patterns.add<LowerSimtVecFuncPattern>(ctx, module, nextSimtRegionId, loweredMemrefByValue);
    patterns.add<LowerVecFuncRegionPattern>(
        ctx, module, nextVectorRegionId, loweredMemrefByValue, invalidScalarAccessBase);
    patterns.add<LowerCopyPattern>(ctx, loweredMemrefByValue);
    patterns.add<LowerThreadIdxPattern, LowerBlockDimPattern>(ctx);
    // NOTE: no dead-tla-scaffolding DCE here. tla-vector-region lowers ops but
    // deliberately leaves the momentary tensor / ptr-bridge scaffolding and
    // unrealized casts in place; the downstream cleanup pass (tla-finalize-memref)
    // is responsible for DCE'ing them.
}

// Per-core identity queries (block_idx / block_num / sub_block_idx) must be
// computed outside a tla.vec.func and passed in; emitting them inside the vector
// region produces an op the vector backend cannot codegen.
static bool isIllegalVecFuncArchOp(Operation* op, StringRef& dslName)
{
    if (isa<::tla::BlockIdxOp>(op)) {
        dslName = "tla.arch.block_idx";
        return true;
    }
    if (isa<::tla::BlockNumOp>(op)) {
        dslName = "tla.arch.block_num";
        return true;
    }
    if (isa<::tla::SubBlockIdxOp>(op)) {
        dslName = "tla.arch.sub_block_idx";
        return true;
    }
    return false;
}

// Validate what a SIMT region is allowed to touch, once per function and before
// the rewrite driver runs -- a pattern would re-report on every retry, and both
// of these are diagnosed far more clearly here than by the LLVM assertions
// hivmc-a5 would otherwise raise. Notes are dropped by the frontend's
// diagnostic handler, so each message is self-contained.
// Rank-2 index folding, run once per function *before* the rewrite driver.
// It must not happen inside a pattern: mutating the tla.vec.func body with a
// plain builder while the greedy driver is matching on it leaves freed ops on
// the driver's worklist, which segfaults rather than failing cleanly.
static LogicalResult linearizeSimtAccesses(func::FuncOp funcOp)
{
    LogicalResult result = success();
    funcOp.walk([&](::tla::VecFuncOp vecFuncOp) {
        if (!isSimtVecFunc(vecFuncOp))
            return;
        if (failed(linearizeSimtAccessesIn(vecFuncOp)))
            result = failure();
    });
    return result;
}

static LogicalResult checkSimtRegionCaptures(func::FuncOp funcOp)
{
    LogicalResult result = success();
    funcOp.walk([&](::tla::VecFuncOp vecFuncOp) {
        if (!isSimtVecFunc(vecFuncOp))
            return;
        SimtCaptures captures;
        collectSimtCaptures(vecFuncOp, captures);

        // Gather every offender first: one diagnostic per region listing them all
        // beats one per tensor, especially since the frontend runs the pipeline
        // twice (typed bridge, then CLI fallback) and doubles whatever is emitted.
        SmallVector<std::string, 4> nonGm;
        SmallVector<unsigned, 4> dynamic;
        for (auto [operandIndex, tensor] : llvm::enumerate(captures.tensors)) {
            auto descOp = tensor.getDefiningOp<::tla::TensorDescOp>();
            if (!descOp)
                continue;
            FailureOr<TensorDescriptor> desc = descriptorFromTensorDescOp(descOp);
            if (failed(desc))
                continue;
            if (!simtAddressSpaceFor(desc->addrspace))
                nonGm.push_back((Twine("#") + Twine(operandIndex + 1) + " in '" + desc->addrspace + "'").str());
            else if (!getConstantIntValue(desc->shape[1]))
                dynamic.push_back(operandIndex + 1);
        }

        auto listTensors = [&](ArrayRef<unsigned> indices) {
            std::string text;
            llvm::raw_string_ostream os(text);
            llvm::interleaveComma(indices, os, [&](unsigned index) { os << "#" << index; });
            return os.str();
        };

        // A SIMT vector function is launched with GM pointers only. A UB operand
        // never gets a base established and silently reads as zeros, so refuse it
        // rather than let the kernel produce wrong answers.
        if (!nonGm.empty()) {
            vecFuncOp.emitOpError() << "a SIMT tla.vec.func can only address GM or UB, but " << nonGm.size() << " of "
                                    << captures.tensors.size()
                                    << " tensor(s) used by this region are not: " << llvm::join(nonGm, ", ")
                                    << ". Buffers reach an outlined SIMT vector function as raw pointers, and "
                                       "only the GM and UB address spaces have a SIMT pointer form";
            result = failure();
        }
        // One pointer per buffer crosses the launch ABI, so the parameter must be a
        // statically shaped memref: a dynamic extent lowers to a 5-field descriptor
        // and hivmc-a5 aborts with an opaque LLVM assertion.
        if (!dynamic.empty()) {
            vecFuncOp.emitOpError() << "a SIMT tla.vec.func requires statically shaped buffers, but tensor(s) "
                                    << listTensors(dynamic) << " of " << captures.tensors.size()
                                    << " used by this region have a dynamic extent. The launch passes one "
                                       "pointer per buffer, while a dynamic memref lowers to a 5-field "
                                       "descriptor (base, aligned, offset, size, stride), which makes hivmc-a5 "
                                       "fail with \"Calling a function with a bad signature!\". Drop "
                                       "mark_compact_shape_dynamic(...) from the host tensor so its extent is "
                                       "known at compile time";
            result = failure();
        }
    });
    return result;
}

// Fail compilation if any per-core identity query is used inside a tla.vec.func.
static LogicalResult checkNoArchOpsInVecFunc(func::FuncOp funcOp)
{
    LogicalResult result = success();
    funcOp.walk([&](::tla::VecFuncOp vecFuncOp) {
        // SIMT regions are per-thread code: the identity queries are exactly how a
        // thread finds its slice of the work, so they are legal (and required)
        // there.
        if (isSimtVecFunc(vecFuncOp))
            return;
        vecFuncOp.getBody().walk([&](Operation* op) {
            StringRef dslName;
            if (isIllegalVecFuncArchOp(op, dslName)) {
                op->emitOpError() << "'" << dslName
                                  << "' is not allowed inside a tla.vec.func region; compute it "
                                     "outside the region and pass the value in";
                result = failure();
            }
        });
    });
    return result;
}

class TlaVectorRegionPass : public PassWrapper<TlaVectorRegionPass, OperationPass<ModuleOp>> {
public:
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TlaVectorRegionPass)

    StringRef getArgument() const override
    {
        return "tla-vector-region";
    }
    StringRef getName() const override
    {
        return "TlaVectorRegionPass";
    }
    StringRef getDescription() const override
    {
        return "Outline tla.vector regions and lower fragment ops to vector IR.";
    }

    void getDependentDialects(DialectRegistry& registry) const override
    {
        registry.insert<
            arith::ArithDialect, func::FuncDialect, mlir::memref::MemRefDialect, hivm::HIVMDialect, hivmave::AVEDialect,
            vector::VectorDialect, hivm_regbaseintrins::HIVMRegbaseIntrinsDialect, LLVM::LLVMDialect, math::MathDialect,
            ::tla::TlaDialect>();
    }

    void runOnOperation() override
    {
        ModuleOp module = getOperation();

        nextVectorRegionId = 0;
        nextSimtRegionId = 0;

        // Snapshot the functions up front: lowering a vec.func appends a new
        // vector_region helper to the module, and that helper must not be fed back
        // through the lowering/folding driver (it already holds lowered AVE ops and
        // the carried scf control flow).
        SmallVector<func::FuncOp, 4> funcOps(module.getOps<func::FuncOp>());
        for (func::FuncOp funcOp : funcOps) {
            if (funcOp.isDeclaration())
                continue;
            // Skip the generated vector_region helpers: they already hold lowered AVE
            // ops and the carried scf control flow, and must not be re-driven.
            if (funcOp->hasAttr(kHivmVectorFunctionAttrName))
                continue;
            // Only AIV (and not-yet-split MIX) functions hold vector work. Their core
            // kind is the func_core_type stamped by tla-lower-func (pure-vector
            // entries retain func_core_type = AIV).
            std::optional<hivm::TFuncCoreType> coreType = getFunctionCoreType(funcOp.getOperation());
            if (coreType != hivm::TFuncCoreType::AIV && coreType != hivm::TFuncCoreType::MIX)
                continue;
            if (failed(checkNoArchOpsInVecFunc(funcOp)) || failed(checkSimtRegionCaptures(funcOp)) ||
                failed(linearizeSimtAccesses(funcOp))) {
                signalPassFailure();
                return;
            }
            inlineVectorRegionWrappers(funcOp);
            // Fresh per-function lowering state: the base-memref handoff cache shared
            // by LowerCopyPattern (gm<->ub / ub->l1 cifax runtime calls) and the
            // vec.func helper operand materialization.
            ::tla::TlaTensorMemrefLowering lowering;
            RewritePatternSet patterns(&getContext());
            bool invalidScalarAccessBase = false;
            populateTlaToVectorPatterns(
                patterns, module, nextVectorRegionId, nextSimtRegionId, lowering.loweredMemrefByValue,
                invalidScalarAccessBase);
            if (failed(mlir::applyPatternsGreedily(funcOp, std::move(patterns)))) {
                signalPassFailure();
                return;
            }
            if (invalidScalarAccessBase) {
                signalPassFailure();
                return;
            }
        }

        // A launched SIMT vector function makes the entry a regbase SIMT kernel as
        // well as a SIMD one; the two sets of entry attributes coexist.
        if (nextSimtRegionId > 0) {
            MLIRContext* ctx = &getContext();
            StringRef entryAttr = hacc::stringifyEnum(hacc::HACCToLLVMIRTranslateAttr::ENTRY);
            for (func::FuncOp funcOp : module.getOps<func::FuncOp>()) {
                if (!funcOp->hasAttr(entryAttr))
                    continue;
                funcOp->setAttr(hivm_regbaseintrins::kDavinciKernelAttrName, UnitAttr::get(ctx));
                setC310RegbaseTargetAttr(funcOp.getOperation(), ctx);
            }
        }

        // `module.getOps<func::FuncOp>()` visits only direct functions. Inline any
        // wrapper in nested modules so finalize never sees frontend scaffolding.
        SmallVector<::tla::VectorOp, 4> leftover;
        module.walk([&](::tla::VectorOp vectorOp) { leftover.push_back(vectorOp); });
        IRRewriter rewriter(module.getContext());
        for (::tla::VectorOp vectorOp : leftover) {
            if (!vectorOp)
                continue;
            if (vectorOp->getNumRegions() == 0 || vectorOp.getBody().empty()) {
                rewriter.eraseOp(vectorOp);
                continue;
            }
            Block* body = &vectorOp.getBody().front();
            rewriter.inlineBlockBefore(body, vectorOp->getBlock(), vectorOp->getIterator());
            rewriter.eraseOp(vectorOp);
        }

        // Barrier-only vec.func regions are inlined rather than outlined, and raw
        // TLAIR may place local barriers directly in a vector wrapper. Lower those
        // remaining operations here as well so this pass is the single owner of
        // tla.local_mem_bar lowering.
        SmallVector<::tla::LocalMemBarOp, 4> localMemBars;
        module.walk([&](::tla::LocalMemBarOp op) { localMemBars.push_back(op); });
        for (::tla::LocalMemBarOp op : localMemBars) {
            if (!op->getBlock())
                continue;
            rewriter.setInsertionPoint(op);
            if (failed(lowerLocalMemBar(rewriter, op))) {
                signalPassFailure();
                return;
            }
            rewriter.eraseOp(op);
        }
    }

private:
    int nextVectorRegionId = 0;
    int nextSimtRegionId = 0;
};

} // namespace

std::unique_ptr<Pass> createTlaVectorRegionPass()
{
    return std::make_unique<TlaVectorRegionPass>();
}

void registerTlaVectorRegionPass()
{
    PassRegistration<TlaVectorRegionPass>();
}

} // namespace tla
