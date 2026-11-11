#include "PassesCommon.h"
#include "PassesInternal.h"
#include "Passes/TlaDebugPrint.h"
#include "Passes/TlaTensorToMemref.h"
#include "Passes/TlaVectorRegionCommon.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/HIVMAVE/IR/HIVMAVE.h"
#include "bishengir/Dialect/Utils/Util.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/ADT/TypeSwitch.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include <algorithm>
#include <cstdlib>
#include <functional>
#include <limits>
#include <optional>
#include <tuple>
#include <utility>

namespace tla {
using namespace ::tla::vecregion;
namespace {

/// Align PostMode UPDATE: emit HIVM ``.post`` load/store (UnAlign is not this path).
static Type getUbPtrType(MLIRContext* ctx)
{
    return LLVM::LLVMPointerType::get(ctx, /*addressSpace=*/6);
}

static Value intToUbPtr(OpBuilder& b, Location loc, Value addrI64)
{
    Value asI64 = castValueToI64(b, loc, addrI64);
    return b.create<LLVM::IntToPtrOp>(loc, getUbPtrType(b.getContext()), asI64);
}

/// AscendC ``vlds/vsts`` POST stride is elements; HIVM ``.post`` offset is bytes.
static Value strideElemsToByteOffI32(OpBuilder& b, Location loc, Value strideElems, Type elemType)
{
    int64_t elemBytes = getByteSizeOfFixedWidthScalarType(elemType);
    Value s = ::tla::castToI32(b, loc, strideElems);
    if (elemBytes == 1)
        return s;
    Value scale = b.create<arith::ConstantIntOp>(loc, elemBytes, /*width=*/32);
    return b.create<arith::MulIOp>(loc, s, scale);
}

/// HIVM store Dist for elem type: f32→2 (NORM_B32), f16/bf16→1 (NORM_B16).
static int32_t hivmStoreDistForElem(Type elemType)
{
    if (elemType.isF32())
        return 2;
    if (elemType.isF16() || elemType.isBF16())
        return 1;
    return 0;
}

/// Cast ``newPtr`` to the live UB pointer type without UnrealizedConversionCast
/// (those survive into the VF and tank VLOOP / post-update recognition).
static Value castToLiveUbPtr(OpBuilder& b, Location loc, Value newPtr, Type ubPtrTy)
{
    if (newPtr.getType() == ubPtrTy)
        return newPtr;
    auto fromTy = dyn_cast<LLVM::LLVMPointerType>(newPtr.getType());
    auto toTy = dyn_cast<LLVM::LLVMPointerType>(ubPtrTy);
    if (fromTy && toTy && fromTy.getAddressSpace() != toTy.getAddressSpace())
        return b.create<LLVM::AddrSpaceCastOp>(loc, ubPtrTy, newPtr);
    return b.create<LLVM::BitcastOp>(loc, ubPtrTy, newPtr);
}

/// Emit ``llvm.hivm.vldsx1.post.*`` and return ``{vec, newPtr}``.
static FailureOr<std::pair<Value, Value>> emitVldsx1Post(
    OpBuilder& b, Location loc, Value ptr, Value strideElems, Type elemType, Type vecTy)
{
    Value byteOff = strideElemsToByteOffI32(b, loc, strideElems, elemType);
    Value dist = b.create<arith::ConstantIntOp>(loc, /*value=*/0, /*width=*/32);
    Value postEn = b.create<arith::ConstantIntOp>(loc, /*value=*/1, /*width=*/32);
    Type ubPtrTy = getUbPtrType(b.getContext());
    Type structTy = LLVM::LLVMStructType::getLiteral(b.getContext(), {vecTy, ubPtrTy});
    Operation* op = nullptr;
    if (elemType.isF32())
        op = b.create<hivm_regbaseintrins::Vldsx1PostV64F32InstrOp>(loc, structTy, ptr, byteOff, dist, postEn);
    else if (elemType.isF16())
        op = b.create<hivm_regbaseintrins::Vldsx1PostV128F16InstrOp>(loc, structTy, ptr, byteOff, dist, postEn);
    else if (elemType.isBF16())
        op = b.create<hivm_regbaseintrins::Vldsx1PostV128BF16InstrOp>(loc, structTy, ptr, byteOff, dist, postEn);
    else
        return failure();
    Value vec = b.create<LLVM::ExtractValueOp>(loc, op->getResult(0), 0);
    Value newPtr = b.create<LLVM::ExtractValueOp>(loc, op->getResult(0), 1);
    newPtr = castToLiveUbPtr(b, loc, newPtr, ubPtrTy);
    return std::make_pair(vec, newPtr);
}

/// Emit Align ``vstsx1.post`` and return HW ``newPtr``.
static FailureOr<Value> emitVstsx1Post(
    OpBuilder& b, Location loc, Value src, Value ptr, Value strideElems, Value preg, Type elemType)
{
    Value byteOff = strideElemsToByteOffI32(b, loc, strideElems, elemType);
    Value dist = b.create<arith::ConstantIntOp>(loc, hivmStoreDistForElem(elemType), /*width=*/32);
    Value postEn = b.create<arith::ConstantIntOp>(loc, /*value=*/1, /*width=*/32);
    Type ubPtrTy = getUbPtrType(b.getContext());
    Operation* op = nullptr;
    if (elemType.isF32())
        op =
            b.create<hivm_regbaseintrins::Vstsx1PostV64F32InstrOp>(loc, ubPtrTy, src, ptr, byteOff, dist, postEn, preg);
    else if (elemType.isF16())
        op = b.create<hivm_regbaseintrins::Vstsx1PostV128F16InstrOp>(
            loc, ubPtrTy, src, ptr, byteOff, dist, postEn, preg);
    else if (elemType.isBF16())
        op = b.create<hivm_regbaseintrins::Vstsx1PostV128BF16InstrOp>(
            loc, ubPtrTy, src, ptr, byteOff, dist, postEn, preg);
    else
        return failure();
    return castToLiveUbPtr(b, loc, op->getResult(0), ubPtrTy);
}

static Type uregVectorType(OpBuilder& b)
{
    return VectorType::get({32}, b.getI8Type());
}

static Type dataVectorType(OpBuilder& b, Type elemType)
{
    if (elemType.isF32())
        return VectorType::get({64}, b.getF32Type());
    if (elemType.isF16())
        return VectorType::get({128}, b.getF16Type());
    if (elemType.isBF16())
        return VectorType::get({128}, b.getBF16Type());
    return {};
}

/// Emit load-side UnAlignReg Pre (AscendC ``LoadUnAlignPre`` / ``vldas``).
/// Self-contained UPDATE: each ``load_unalign_update`` emits its own local Pre.
static FailureOr<Value> emitVldas(OpBuilder& b, Location loc, Value ptr)
{
    auto op = b.create<hivm_regbaseintrins::VldasInstrOp>(loc, uregVectorType(b), ptr);
    return op.getResult();
}

/// Empty store ureg (AscendC ``UnalignRegForStore`` — no ``vldas`` / Pre).
/// HIVM ``hivm.init.vector.align.data``; same as AVE unaligned store path.
static Value emitEmptyStoreUreg(OpBuilder& b, Location loc)
{
    return b.create<hivm_regbaseintrins::InitVectorAlignDataInstrOp>(loc, uregVectorType(b)).getResult();
}

static Value getVectorHelperStorageIdentity(Value source);

/// Emit ``vldus.post`` and return ``{vec, newUreg, newPtr}``.
///
/// ``$cceVL`` is the AscendC ``postUpdateStride`` in **bytes**
/// (``stride_elems × sizeof(T)``, same as Align UPDATE). HW ``newPtr`` is the
/// TLA ``next_ptr`` — no software pointer arithmetic. Each UPDATE stays
/// self-contained (local ``vldas``); ureg is not carried across ops.
static FailureOr<std::tuple<Value, Value, Value>> emitVldusPost(
    OpBuilder& b, Location loc, Value ptr, Value ureg, Value countBytes, Type elemType)
{
    Type vecTy = dataVectorType(b, elemType);
    if (!vecTy)
        return failure();
    Type structTy =
        LLVM::LLVMStructType::getLiteral(b.getContext(), {vecTy, uregVectorType(b), getUbPtrType(b.getContext())});
    Operation* op = nullptr;
    if (elemType.isF32())
        op = b.create<hivm_regbaseintrins::VldusPostF32InstrOp>(loc, structTy, ptr, ureg, countBytes);
    else if (elemType.isF16())
        op = b.create<hivm_regbaseintrins::VldusPostF16InstrOp>(loc, structTy, ptr, ureg, countBytes);
    else if (elemType.isBF16())
        op = b.create<hivm_regbaseintrins::VldusPostBF16InstrOp>(loc, structTy, ptr, ureg, countBytes);
    else
        return failure();
    Value vec = b.create<LLVM::ExtractValueOp>(loc, op->getResult(0), 0);
    Value newUreg = b.create<LLVM::ExtractValueOp>(loc, op->getResult(0), 1);
    Value newPtr =
        castToLiveUbPtr(b, loc, b.create<LLVM::ExtractValueOp>(loc, op->getResult(0), 2), getUbPtrType(b.getContext()));
    return std::make_tuple(vec, newUreg, newPtr);
}

/// Emit ``vstus.post`` and return ``{newUreg, newPtr}``.
///
/// ``$eleSize`` is ``postUpdateStride`` in bytes (same rule as ``emitVldusPost``).
static FailureOr<std::pair<Value, Value>> emitVstusPost(
    OpBuilder& b, Location loc, Value src, Value ptr, Value ureg, Value countBytes, Type elemType)
{
    Type structTy = LLVM::LLVMStructType::getLiteral(b.getContext(), {uregVectorType(b), getUbPtrType(b.getContext())});
    Operation* op = nullptr;
    if (elemType.isF32())
        op = b.create<hivm_regbaseintrins::VstusPostF32InstrOp>(loc, structTy, src, ptr, countBytes, ureg);
    else if (elemType.isF16())
        op = b.create<hivm_regbaseintrins::VstusPostF16InstrOp>(loc, structTy, src, ptr, countBytes, ureg);
    else if (elemType.isBF16())
        op = b.create<hivm_regbaseintrins::VstusPostBF16InstrOp>(loc, structTy, src, ptr, countBytes, ureg);
    else
        return failure();
    Value newUreg = b.create<LLVM::ExtractValueOp>(loc, op->getResult(0), 0);
    Value newPtr =
        castToLiveUbPtr(b, loc, b.create<LLVM::ExtractValueOp>(loc, op->getResult(0), 1), getUbPtrType(b.getContext()));
    return std::make_pair(newUreg, newPtr);
}

static LogicalResult emitVstas(OpBuilder& b, Location loc, Value ureg, Value ptr)
{
    // ``hivm.vstas`` args: (data, base, offset, postUpdateEnable).
    // Non-post flush of the align ureg: offset=0, postUpdateEnable=0 (pointer
    // advance already done by ``vstus.post``). Same as AscendC ``vstar`` /
    // ``__builtin_cce_vstas_*(..., 0, 0 /* pre */)`` and HIVM ``buildVstasOp``.
    // Do not pass postUpdateEnable=1 here — that belongs to ``hivm.vstas.post``.
    Value zero = b.create<arith::ConstantIntOp>(loc, /*value=*/0, /*width=*/32);
    b.create<hivm_regbaseintrins::VstasInstrOp>(loc, ureg, ptr, zero, zero);
    return success();
}

static hivmave::VFPgeOp createAvePgeMask(OpBuilder& b, Location loc, VectorType maskType, hivmave::PgePattern pattern)
{
    return b.create<hivmave::VFPgeOp>(loc, maskType, pattern);
}

static hivmave::VFPltOp createAvePltMask(OpBuilder& b, Location loc, VectorType maskType, Value trueShape)
{
    return b.create<hivmave::VFPltOp>(loc, maskType, b.getIndexType(), trueShape);
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
        case ::LoadDist::unpack4_b8:
            return hivmave::LoadDist::UNPK4_B8;
        case ::LoadDist::e2b_b16:
            return hivmave::LoadDist::E2B_B16;
        case ::LoadDist::e2b_b32:
            return hivmave::LoadDist::E2B_B32;
        case ::LoadDist::blk:
            return hivmave::LoadDist::BLK;
        // MaskDist US/DS: IR may carry these enums, but lowering uses AIV bc
        // (NPU-IR i1→plds hardcodes NORM). Keep map entries for completeness.
        case ::LoadDist::us:
            return hivmave::LoadDist::US;
        case ::LoadDist::ds:
            return hivmave::LoadDist::DS;
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
        case ::StoreDist::pack4_b32:
            return hivmave::StoreDist::PK4_B32;
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

// Every tla binary op carries the same three operands, so the tables below read
// them generically: adding an op means adding one row, not editing a second
// parallel switch that fails silently when it is forgotten.
template <typename OpT>
static VectorOpInfo binaryInfo(OpT o, VectorBinaryKind kind, VectorRhsKind rhsKind, StringRef mnemonic)
{
    return VectorOpInfo{kind, rhsKind, mnemonic, TlaBinaryOperands{o.getLhs(), o.getRhs(), o.getMask()}};
}

/// Binary ops whose rhs is a whole vector.
static std::optional<VectorOpInfo> getVectorBinaryInfo(Operation* op)
{
    if (!op)
        return std::nullopt;
    using K = VectorBinaryKind;
    constexpr VectorRhsKind kVec = VectorRhsKind::Vector;
    return TypeSwitch<Operation*, std::optional<VectorOpInfo>>(op)
        .Case([](::tla::AddOp o) { return binaryInfo(o, K::Add, kVec, "add"); })
        .Case([](::tla::SubOp o) { return binaryInfo(o, K::Sub, kVec, "sub"); })
        .Case([](::tla::MulOp o) { return binaryInfo(o, K::Mul, kVec, "mul"); })
        .Case([](::tla::DivOp o) { return binaryInfo(o, K::Div, kVec, "div"); })
        .Case([](::tla::MaxOp o) { return binaryInfo(o, K::Max, kVec, "max"); })
        .Case([](::tla::MinOp o) { return binaryInfo(o, K::Min, kVec, "min"); })
        .Case([](::tla::BitwiseAndOp o) { return binaryInfo(o, K::And, kVec, "bitwise_and"); })
        .Case([](::tla::BitwiseOrOp o) { return binaryInfo(o, K::Or, kVec, "bitwise_or"); })
        .Case([](::tla::BitwiseXorOp o) { return binaryInfo(o, K::Xor, kVec, "bitwise_xor"); })
        .Case([](::tla::ShiftLeftOp o) { return binaryInfo(o, K::ShiftLeft, kVec, "shift_left"); })
        .Case([](::tla::ShiftRightOp o) { return binaryInfo(o, K::ShiftRight, kVec, "shift_right"); })
        .Default(std::optional<VectorOpInfo>{std::nullopt});
}

/// Binary ops whose rhs is a scalar splat across the vector.
static std::optional<VectorOpInfo> getVectorScalarBinaryInfo(Operation* op)
{
    if (!op)
        return std::nullopt;
    using K = VectorBinaryKind;
    constexpr VectorRhsKind kScalar = VectorRhsKind::Scalar;
    return TypeSwitch<Operation*, std::optional<VectorOpInfo>>(op)
        .Case([](::tla::AddsOp o) { return binaryInfo(o, K::Add, kScalar, "adds"); })
        .Case([](::tla::SubsOp o) { return binaryInfo(o, K::Sub, kScalar, "subs"); })
        .Case([](::tla::MulsOp o) { return binaryInfo(o, K::Mul, kScalar, "muls"); })
        .Case([](::tla::MaxsOp o) { return binaryInfo(o, K::Max, kScalar, "maxs"); })
        .Case([](::tla::MinsOp o) { return binaryInfo(o, K::Min, kScalar, "mins"); })
        .Case([](::tla::DivsOp o) { return binaryInfo(o, K::Div, kScalar, "divs"); })
        .Case([](::tla::ShiftLeftsOp o) { return binaryInfo(o, K::ShiftLeft, kScalar, "shift_lefts"); })
        .Case([](::tla::ShiftRightsOp o) { return binaryInfo(o, K::ShiftRight, kScalar, "shift_rights"); })
        .Default(std::optional<VectorOpInfo>{std::nullopt});
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
           isa_and_nonnull<::tla::HistogramsOp>(op) || isa_and_nonnull<::tla::ReduceOp>(op) ||
           isa_and_nonnull<::tla::ReduceIntraBlockOp>(op) || isa_and_nonnull<::tla::GatherOp>(op) ||
           isa_and_nonnull<::tla::CastOp>(op) || isa_and_nonnull<::tla::BitCastOp>(op) ||
           isa_and_nonnull<::tla::PackOp>(op) || isa_and_nonnull<::tla::UnpackOp>(op) ||
           isa_and_nonnull<::tla::InterleaveOp>(op) || isa_and_nonnull<::tla::DeInterleaveOp>(op);
}

static void annotateAivLibraryCall(func::FuncOp callee)
{
    MLIRContext* ctx = callee.getContext();
    callee.setPrivate();
    callee->setAttr("llvm.emit_c_interface", UnitAttr::get(ctx));
    callee->setAttr(hivm::TFuncCoreTypeAttr::name, hivm::TFuncCoreTypeAttr::get(ctx, hivm::TFuncCoreType::AIV));
}

// MaskLoadDist US/DS: one bc symbol per UB element byte width. Element type
// is address typing only; plds always takes __ubuf__ uint32_t*. Typed stubs
// keep the memref ABI exact (PointerCast-to-i8 views have tripped BiShengLIR).
static StringRef maskLoadUsDsCalleeName(int64_t elemBytes, hivmave::LoadDist pattern)
{
    const bool isUs = pattern == hivmave::LoadDist::US;
    switch (elemBytes) {
        case 1:
            return isUs ? "mask_load_us_b8" : "mask_load_ds_b8";
        case 2:
            return isUs ? "mask_load_us_b16" : "mask_load_ds_b16";
        case 4:
            return isUs ? "mask_load_us_b32" : "mask_load_ds_b32";
        case 8:
            return isUs ? "mask_load_us_b64" : "mask_load_ds_b64";
        default:
            return {};
    }
}

static func::FuncOp getOrCreateMaskLoadUsDsLibraryCall(
    ModuleOp module, Location loc, Type memRefType, hivmave::LoadDist pattern)
{
    auto memrefTy = dyn_cast<MemRefType>(memRefType);
    if (!memrefTy)
        return {};
    int64_t elemBytes = getByteSizeOfFixedWidthScalarType(memrefTy.getElementType());
    if (elemBytes != 1 && elemBytes != 2 && elemBytes != 4 && elemBytes != 8)
        return {};
    if (pattern != hivmave::LoadDist::US && pattern != hivmave::LoadDist::DS)
        return {};
    StringRef calleeName = maskLoadUsDsCalleeName(elemBytes, pattern);
    if (calleeName.empty())
        return {};
    if (auto existing = module.lookupSymbol<func::FuncOp>(calleeName)) {
        annotateAivLibraryCall(existing);
        return existing;
    }
    OpBuilder moduleBuilder(module.getBodyRegion());
    auto pregTy = VectorType::get({256}, IntegerType::get(module.getContext(), 1));
    auto fnType = FunctionType::get(module.getContext(), {memRefType}, {pregTy});
    auto callee = moduleBuilder.create<func::FuncOp>(loc, calleeName, fnType);
    annotateAivLibraryCall(callee);
    return callee;
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

static std::string getVectorPackLibraryCallName(Type srcElement, Type dstElement, ::RegPart part)
{
    auto src = cast<IntegerType>(srcElement);
    auto dst = cast<IntegerType>(dstElement);
    return std::string("pack_reg_uint") + std::to_string(dst.getWidth()) + "_t_" + (src.isUnsigned() ? "uint" : "int") +
           std::to_string(src.getWidth()) + "_t_" + (part == ::RegPart::high ? "high" : "low");
}

static std::string getVectorUnpackLibraryCallName(Type srcElement, Type dstElement, ::RegPart part)
{
    auto src = cast<IntegerType>(srcElement);
    auto dst = cast<IntegerType>(dstElement);
    return std::string("unpack_reg_") + (dst.isUnsigned() ? "uint" : "int") + std::to_string(dst.getWidth()) + "_t_" +
           (src.isUnsigned() ? "uint" : "int") + std::to_string(src.getWidth()) + "_t_" +
           (part == ::RegPart::high ? "high" : "low");
}

static func::FuncOp getOrCreateVectorPackUnpackLibraryCall(
    ModuleOp module, Location loc, VectorType srcType, VectorType dstType, StringRef name)
{
    if (auto existing = module.lookupSymbol<func::FuncOp>(name))
        return existing;
    OpBuilder moduleBuilder(module.getBodyRegion());
    auto type = FunctionType::get(module.getContext(), {srcType}, {dstType});
    auto callee = moduleBuilder.create<func::FuncOp>(loc, name, type);
    annotateAivLibraryCall(callee);
    return callee;
}

static std::string getGatherRegLibraryCallName(Type elementType)
{
    if (elementType.isF32())
        return "gather_reg_float";
    if (elementType.isF16())
        return "gather_reg_half";
    if (elementType.isBF16())
        return "gather_reg_bfloat16_t";
    if (auto intType = dyn_cast<IntegerType>(elementType)) {
        const unsigned w = intType.getWidth();
        const bool uns = intType.isUnsigned();
        if (w == 8)
            return uns ? "gather_reg_uint8_t" : "gather_reg_int8_t";
        if (w == 16)
            return uns ? "gather_reg_uint16_t" : "gather_reg_int16_t";
        if (w == 32)
            return uns ? "gather_reg_uint32_t" : "gather_reg_int32_t";
    }
    return {};
}

// mode / type mirror HistogramsParams / AscendC (BIN half vs FREQUENCY|ACCUMULATE).
static std::string getHistogramsLibraryCallName(HistogramsBin mode, HistogramsType type)
{
    const bool accumulate = type == HistogramsType::accumulate;
    const bool bin1 = mode == HistogramsBin::bin1;
    if (!accumulate && !bin1)
        return "histograms_frequency_bin0_u8";
    if (!accumulate && bin1)
        return "histograms_frequency_bin1_u8";
    if (accumulate && !bin1)
        return "histograms_accumulate_bin0_u8";
    return "histograms_accumulate_bin1_u8";
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

static func::FuncOp getOrCreateGatherRegLibraryCall(
    ModuleOp module, Location loc, VectorType srcVecType, VectorType indexVecType, StringRef calleeName)
{
    // Mixed AIC/AIV kernels require hivm.func_core_type on linked AIV bitcode
    // stubs; without it HIVMC reports "Unknown core type" for gather_reg_*.
    if (auto existing = module.lookupSymbol<func::FuncOp>(calleeName)) {
        annotateAivLibraryCall(existing);
        return existing;
    }
    OpBuilder moduleBuilder(module.getBodyRegion());
    auto fnType = FunctionType::get(module.getContext(), {srcVecType, indexVecType}, {srcVecType});
    auto callee = moduleBuilder.create<func::FuncOp>(loc, calleeName, fnType);
    annotateAivLibraryCall(callee);
    return callee;
}

static func::FuncOp getOrCreateHistogramsLibraryCall(
    ModuleOp module, Location loc, VectorType srcVecType, VectorType dstVecType, VectorType pregType,
    StringRef calleeName)
{
    if (auto existing = module.lookupSymbol<func::FuncOp>(calleeName))
        return existing;
    OpBuilder moduleBuilder(module.getBodyRegion());
    auto fnType = FunctionType::get(module.getContext(), {srcVecType, dstVecType, pregType}, {dstVecType});
    auto callee = moduleBuilder.create<func::FuncOp>(loc, calleeName, fnType);
    callee.setPrivate();
    callee->setAttr("llvm.emit_c_interface", UnitAttr::get(module.getContext()));
    return callee;
}

static func::FuncOp getOrCreateMaskSelectLibraryCall(ModuleOp module, Location loc, VectorType pregType)
{
    constexpr llvm::StringLiteral calleeName = "tla_mask_select";
    if (auto existing = module.lookupSymbol<func::FuncOp>(calleeName))
        return existing;
    OpBuilder moduleBuilder(module.getBodyRegion());
    auto fnType = FunctionType::get(module.getContext(), {pregType, pregType, pregType}, {pregType});
    auto callee = moduleBuilder.create<func::FuncOp>(loc, calleeName, fnType);
    callee.setPrivate();
    callee->setAttr("llvm.emit_c_interface", UnitAttr::get(module.getContext()));
    return callee;
}

/// Shared bc type suffix for DATA_BLOCK_COPY helpers in ``bc/Vector/{load,store}.cpp``.
static std::string dataBlockCopyElementSuffix(Type elementType)
{
    if (elementType.isF32())
        return "float";
    if (elementType.isF16())
        return "half";
    if (elementType.isBF16())
        return "bf16";
    if (elementType.isInteger(32))
        return elementType.isUnsignedInteger() ? "uint32" : "int32";
    if (elementType.isInteger(16))
        return elementType.isUnsignedInteger() ? "uint16" : "int16";
    if (isa<Float8E4M3FNType>(elementType))
        return "fp8_e4m3fn";
    if (isa<Float8E5M2Type>(elementType))
        return "fp8_e5m2";
    if (elementType.isInteger(8) && !elementType.isUnsignedInteger())
        return "int8";
    return {};
}

static std::string getStoreWithStrideLibraryCallName(Type elementType)
{
    std::string suffix = dataBlockCopyElementSuffix(elementType);
    return suffix.empty() ? std::string{} : ("store_with_stride_" + suffix);
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

static std::string getLoadWithStrideLibraryCallName(Type elementType)
{
    std::string suffix = dataBlockCopyElementSuffix(elementType);
    return suffix.empty() ? std::string{} : ("load_with_stride_" + suffix);
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

// vsldb.post / vsstb.post are not HIVMC ops. The bc helper returns the hardware
// pointer (store) or writes it to an i64 slot (load, so the VectorReg return
// ABI stays the single-result form already used by load_with_stride).
// Helpers already use the ``_mlir_ciface_*`` ABI in ``bc/Vector/{load,store}.cpp``
// (raw UB pointer, not a memref descriptor). Call that symbol directly.
// ``llvm.emit_c_interface`` would keep the bare name as the simd_vf callee,
// and bisheng then fails to find a body to inline.
static void annotateDataBlockCopyUpdateLibraryCall(func::FuncOp callee)
{
    MLIRContext* ctx = callee.getContext();
    callee.setPrivate();
    callee->setAttr(hivm::TFuncCoreTypeAttr::name, hivm::TFuncCoreTypeAttr::get(ctx, hivm::TFuncCoreType::AIV));
}

static func::FuncOp getOrCreateLoadUpdateRetptrLibraryCall(
    ModuleOp module, Location loc, Type vecType, StringRef calleeName)
{
    auto ctx = module.getContext();
    if (auto existing = module.lookupSymbol<func::FuncOp>(calleeName)) {
        annotateDataBlockCopyUpdateLibraryCall(existing);
        return existing;
    }
    OpBuilder moduleBuilder(module.getBodyRegion());
    auto ubPtr = getUbPtrType(ctx);
    auto slotPtr = LLVM::LLVMPointerType::get(ctx);
    auto fnType = FunctionType::get(
        ctx,
        {ubPtr, IntegerType::get(ctx, 32), IntegerType::get(ctx, 32), VectorType::get({256}, IntegerType::get(ctx, 1)),
         slotPtr},
        {vecType});
    auto callee = moduleBuilder.create<func::FuncOp>(loc, calleeName, fnType);
    annotateDataBlockCopyUpdateLibraryCall(callee);
    return callee;
}

static func::FuncOp getOrCreateStoreUpdateRetptrLibraryCall(
    ModuleOp module, Location loc, Type vecType, StringRef calleeName)
{
    auto ctx = module.getContext();
    if (auto existing = module.lookupSymbol<func::FuncOp>(calleeName)) {
        annotateDataBlockCopyUpdateLibraryCall(existing);
        return existing;
    }
    OpBuilder moduleBuilder(module.getBodyRegion());
    auto ubPtr = getUbPtrType(ctx);
    auto fnType = FunctionType::get(
        ctx,
        {vecType, ubPtr, IntegerType::get(ctx, 32), IntegerType::get(ctx, 32),
         VectorType::get({256}, IntegerType::get(ctx, 1))},
        {ubPtr});
    auto callee = moduleBuilder.create<func::FuncOp>(loc, calleeName, fnType);
    annotateDataBlockCopyUpdateLibraryCall(callee);
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

// AVE vectors are signless; peel unsigned from the tla operand when present.
static Type tlaElementOr(Type tlaType, Type fallback)
{
    if (auto v = dyn_cast<::tla::VectorSSAType>(tlaType))
        return v.getElementType();
    return fallback;
}

static bool isUnsignedInt(Type type)
{
    auto i = dyn_cast<IntegerType>(type);
    return i && i.isUnsigned();
}

// Build the AVE vector op for a tla binary op. The mask controls active lanes.
// Div uses TypeFn for signedness; unsigned max/min select vumax/vumin.
static Value createVectorBinaryResult(
    OpBuilder& b, Location loc, VectorBinaryKind kind, Type tlaOperandType, Type elementType, VectorType vecType,
    Value lhs, Value rhs, Value mask)
{
    const bool isUnsigned = isUnsignedInt(tlaElementOr(tlaOperandType, elementType));
    switch (kind) {
        case VectorBinaryKind::Add:
            return b.create<hivmave::VFAddOp>(loc, vecType, lhs, rhs, mask).getResult();
        case VectorBinaryKind::Sub:
            return b.create<hivmave::VFSubOp>(loc, vecType, lhs, rhs, mask).getResult();
        case VectorBinaryKind::Mul:
            return b.create<hivmave::VFMulOp>(loc, vecType, lhs, rhs, mask).getResult();
        case VectorBinaryKind::Div: {
            auto cast = isUnsigned ? hivm::TypeFn::cast_unsigned : hivm::TypeFn::cast_signed;
            return b.create<hivmave::VFDivOp>(loc, vecType, lhs, rhs, mask, hivm::TypeFnAttr::get(b.getContext(), cast))
                .getResult();
        }
        case VectorBinaryKind::Max:
            if (isUnsigned)
                return b.create<hivmave::VMaxUIOp>(loc, vecType, lhs, rhs, mask).getResult();
            return b.create<hivmave::VFMaxOp>(loc, vecType, lhs, rhs, mask).getResult();
        case VectorBinaryKind::Min:
            if (isUnsigned)
                return b.create<hivmave::VMinUIOp>(loc, vecType, lhs, rhs, mask).getResult();
            return b.create<hivmave::VFMinOp>(loc, vecType, lhs, rhs, mask).getResult();
        case VectorBinaryKind::And:
            if (isa<::tla::VectorSSAType>(tlaOperandType))
                return b.create<hivmave::VFAndOp>(loc, vecType, lhs, rhs, mask).getResult();
            if (isa<::tla::MaskSSAType>(tlaOperandType))
                return b
                    .create<hivmave::PregAndOp>(
                        loc, vecType, maskWidthAttrForMaskType(b, cast<::tla::MaskSSAType>(tlaOperandType)), lhs, rhs,
                        mask)
                    .getRes();
            return nullptr;
        case VectorBinaryKind::Or:
            if (isa<::tla::VectorSSAType>(tlaOperandType))
                return b.create<hivmave::VFOrOp>(loc, vecType, lhs, rhs, mask).getResult();
            if (isa<::tla::MaskSSAType>(tlaOperandType))
                return b
                    .create<hivmave::PregOrOp>(
                        loc, vecType, maskWidthAttrForMaskType(b, cast<::tla::MaskSSAType>(tlaOperandType)), lhs, rhs,
                        mask)
                    .getRes();
            return nullptr;
        case VectorBinaryKind::Xor:
            if (isa<::tla::VectorSSAType>(tlaOperandType))
                return b.create<hivmave::VFXorOp>(loc, vecType, lhs, rhs, mask).getResult();
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
//
// The bit width is carried by functionType (hivmave::FunctionDistType, i.e.
// #ave.func_dist_type<pb8|pb16|pb32>). Without it, readers fall back to the
// container width (VL_BITS / predicate size), so even an f32 operand gets a b8
// mask that covers the wrong elements.
static constexpr llvm::StringLiteral kFunctionType = "functionType";
static void annotateFullPregWidth(OpBuilder& b, Operation* op, VectorType resultType, int64_t semanticLanes)
{
    if (resultType.getNumElements() == semanticLanes)
        return;
    const int64_t bitWidth = maskElementBitWidthForLanes(semanticLanes);
    hivmave::FunctionDistType dist = bitWidth >= 32 ? hivmave::FunctionDistType::PB32 :
                                     bitWidth >= 16 ? hivmave::FunctionDistType::PB16 :
                                                      hivmave::FunctionDistType::PB8;
    op->setAttr(kFunctionType, hivmave::FunctionDistTypeAttr::get(b.getContext(), dist));
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

    /// i64 out-param slot for load_update retptr helpers. Always at the outlined
    /// helper entry — not the enclosing ``scf.for`` / ``scf.if`` region.
    /// ``lowerScfFor`` body callbacks run before the for-region is linked, so
    /// ``getInsertionBlock()->getParentOp()`` is null there; an escaped alloca
    /// left in the loop body would grow the AIV stack every trip after scf→cf.
    FailureOr<Value> createI64RetptrSlot(OpBuilder& b, Location loc)
    {
        if (!entry)
            return failure();
        OpBuilder::InsertionGuard guard(b);
        b.setInsertionPointToStart(entry);
        Value one = b.create<LLVM::ConstantOp>(loc, b.getI32Type(), b.getI32IntegerAttr(1));
        Type slotPtrTy = LLVM::LLVMPointerType::get(b.getContext());
        return b.create<LLVM::AllocaOp>(loc, slotPtrTy, b.getI64Type(), one).getResult();
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
// CAST_NONE matches AscendC: when a path actually consumes rnd (lossy cast /
// same-type vtrc / etc.) it becomes RINT; lossless widen paths such as
// vextf ignore the attribute entirely and never call this for effect.
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
        case ::RoundMode::cast_none:
        case ::RoundMode::cast_rint:
            hv = hivm::RoundMode::RINT;
            break;
        case ::RoundMode::cast_odd:
            hv = hivm::RoundMode::ODD;
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
    Value shifted = b.create<hivmave::VFShlsOp>(loc, dstBitsType, widened, shift, mask, sign).getResult();
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
        if (i.getWidth() == 1) // bool
            return false;
        unsigned w = i.getWidth();
        return w == 8 || w == 16 || w == 32 || w == 64;
    }
    return false;
}

// Bitcast an integer vector between unsigned and signless of the same width.
// AVE cast ops are typed on signless integers; unsigned VectorSSA stays
// unsigned so Compare can select VCmpU*.
static Value bitcastIntVectorSignedness(OpBuilder& b, Location loc, Value v, Type targetElem)
{
    auto srcTy = cast<VectorType>(v.getType());
    if (srcTy.getElementType() == targetElem)
        return v;
    auto dstTy = VectorType::get(srcTy.getShape(), targetElem);
    return b.create<mlir::vector::BitCastOp>(loc, dstTy, v).getResult();
}

static Type signlessIntOfWidth(MLIRContext* ctx, unsigned width)
{
    return IntegerType::get(ctx, width);
}

// Unsigned cast routes are an explicit short list (frontend
// `_CAST_UNSIGNED_ROUTES`): a missing AVE arm reaches llvm_unreachable.
static FailureOr<Value> createUnsignedVectorCastResult(
    OpBuilder& b, AutoFullPredicateCache& fullMasks, Location loc, VectorType srcVecType, VectorType dstVecType,
    Type tlaSrcElem, Type tlaDstElem, Value src, Value mask, bool hasUserMask, hivmave::VCVT_PartTypeAttr part,
    hivmave::VCVT_PPTypeAttr pp, hivm::RoundModeAttr rnd, BoolAttr sat)
{
    unsigned sb = tlaSrcElem.getIntOrFloatBitWidth();
    unsigned db = tlaDstElem.getIntOrFloatBitWidth();
    const bool sUnsigned = tlaSrcElem.isUnsignedInteger();
    const bool dUnsigned = tlaDstElem.isUnsignedInteger();
    auto nullPart = hivmave::VCVT_PartTypeAttr();
    auto nullPP = hivmave::VCVT_PPTypeAttr();
    MLIRContext* ctx = b.getContext();

    if ((sUnsigned && sb == 64) || (dUnsigned && db == 64))
        return failure();

    Value aveSrc = src;
    VectorType aveSrcTy = srcVecType;
    if (sUnsigned) {
        Type signless = signlessIntOfWidth(ctx, sb);
        aveSrc = bitcastIntVectorSignedness(b, loc, src, signless);
        aveSrcTy = cast<VectorType>(aveSrc.getType());
    }

    // Keep the AVE region signless; signedness lives on the tla type.
    auto wrapDst = [&](Value aveDst) -> Value { return aveDst; };

    if (sUnsigned && sb == 8 && tlaDstElem.isF32()) {
        Value gated = aveSrc;
        Value legMask = mask;
        if (hasUserMask) {
            gated = b.create<hivmave::VFAndOp>(loc, aveSrcTy, aveSrc, aveSrc, mask).getResult();
            legMask = allTrueMaskFor(fullMasks, loc, aveSrcTy, aveSrcTy);
        }
        auto midType = VectorType::get(dstVecType.getShape(), b.getIntegerType(32));
        Value mid = b.create<hivmave::VFExtUIOp>(loc, midType, gated, legMask, nullPart, pp).getResult();
        Value midMask = hasUserMask ? allTrueMaskFor(fullMasks, loc, midType, midType) : legMask;
        return b.create<hivmave::VFSIntToFpOp>(loc, dstVecType, mid, midMask, rnd, nullPart).getResult();
    }

    if (sUnsigned && sb == 8 && tlaDstElem.isF16())
        return b.create<hivmave::VFUIntToFpOp>(loc, dstVecType, aveSrc, mask, hivm::RoundModeAttr(), part).getResult();
    if (tlaSrcElem.isF16() && dUnsigned && db == 8) {
        auto aveDstTy = VectorType::get(dstVecType.getShape(), signlessIntOfWidth(ctx, 8));
        Value aveDst = b.create<hivmave::VFFpToUIntOp>(loc, aveDstTy, src, mask, rnd, sat, part).getResult();
        return wrapDst(aveDst);
    }

    if (!sUnsigned || !dUnsigned)
        return failure();

    auto aveDstTy = VectorType::get(dstVecType.getShape(), signlessIntOfWidth(ctx, db));
    if (db > sb) {
        if (db / sb >= 4)
            return wrapDst(b.create<hivmave::VFExtUIOp>(loc, aveDstTy, aveSrc, mask, nullPart, pp).getResult());
        return wrapDst(b.create<hivmave::VFExtUIOp>(loc, aveDstTy, aveSrc, mask, part, nullPP).getResult());
    }
    if (db == sb)
        return failure();

    auto uni = hivm::UnsignedModeAttr::get(ctx, hivm::UnsignedMode::UI2UI);
    if (sb / db >= 4)
        return wrapDst(b.create<hivmave::VFTruncIOp>(loc, aveDstTy, aveSrc, mask, sat, nullPart, pp, uni).getResult());
    return wrapDst(b.create<hivmave::VFTruncIOp>(loc, aveDstTy, aveSrc, mask, sat, part, nullPP, uni).getResult());
}

// Build the AVE op or regbase intrinsic for a tla.cast, dispatching by (src, dst) element kind.
// The trait supplies rounding, saturation and register layout; the mask (source
// width) predicates active lanes.
static FailureOr<Value> createVectorCastResult(
    OpBuilder& b, AutoFullPredicateCache& fullMasks, Location loc, VectorType srcVecType, VectorType dstVecType,
    Type tlaSrcElem, Type tlaDstElem, ArrayRef<int32_t> trait, Value src, Value mask, bool hasUserMask)
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

    // AVE vextf has no rounding operand and its lowering only produces f32.
    // The direct f16 -> bf16 instruction preserves every lane without a part
    // selector. It has no saturation operand, so only UNKNOWN is accepted.
    if (s.isF16() && d.isBF16()) {
        if (static_cast<::SatMode>(trait[1]) != ::SatMode::unknown)
            return emitError(loc, "f16->bf16 cast only supports UNKNOWN sat_mode"), failure();
        Value preg = castMaskToPregType(b, loc, mask, fullPregVecType(b.getContext()));
        Value round = b.create<arith::ConstantIntOp>(loc, static_cast<int64_t>(rnd.getValue()), 32);
        return b.create<hivm_regbaseintrins::VcvtffF162BF16InstrOp>(loc, dstVecType, src, preg, round).getResult();
    }

    if (tlaSrcElem.isUnsignedInteger() || tlaDstElem.isUnsignedInteger())
        return createUnsignedVectorCastResult(
            b, fullMasks, loc, srcVecType, dstVecType, tlaSrcElem, tlaDstElem, src, mask, hasUserMask, part, pp, rnd,
            sat);

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
            Value gated = b.create<hivmave::VFAndOp>(loc, srcBitsType, srcBits, srcBits, mask).getResult();
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
            Value dstBits = b.create<hivmave::VFOrOp>(loc, dstBitsType, dstEvenBits, dstOddBits, dstMask).getResult();
            return b.create<mlir::vector::BitCastOp>(loc, dstVecType, dstBits).getResult();
        }
    }

    // The native fp8<->f32 instructions select one of four byte positions in
    // each 32-bit slot. Keep every RegSlot value intact as pp0..pp3.
    if (isAveFloat8Type(s) && d.isF32())
        return createFp8ToF32Result(b, loc, dstVecType, src, mask, pp);
    if (s.isF32() && isAveFloat8Type(d))
        return createF32ToFp8Result(b, loc, dstVecType, src, mask, rnd, sat, pp);
    // A same-type float cast has no width to convert, so it is a pure re-round:
    // vtrc applies the round mode to every lane and nothing else. It has no
    // saturation operand and no even/odd part selector, so the trait's sat_mode
    // and reg_slot are simply ignored here (the frontend rejects the values
    // that would silently mean nothing).
    if (sFloat && dFloat && s == d)
        return b.create<hivmave::VFVtrcOp>(loc, dstVecType, src, rnd, mask).getResult();
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
            return b.create<hivmave::VFExpOp>(loc, vecType, operand, mask).getResult();
        case VectorUnaryKind::Log:
            return b.create<hivmave::VFLnOp>(loc, vecType, operand, mask).getResult();
        case VectorUnaryKind::Sqrt:
            return b.create<hivmave::VFSqrtOp>(loc, vecType, operand, mask).getResult();
        case VectorUnaryKind::Abs:
            return b.create<hivmave::VFAbsOp>(loc, vecType, operand, mask).getResult();
        case VectorUnaryKind::Neg:
            return b.create<hivmave::VFNegOp>(loc, vecType, operand, mask).getResult();
        case VectorUnaryKind::Not:
            if (isa<::tla::VectorSSAType>(tlaOperandType))
                return b.create<hivmave::VFNotOp>(loc, vecType, operand, mask).getResult();
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

// AVE rejects ui*; map unsigned tla elements to signless of the same width.
// Signedness for Compare/Cast/Div/Reduce/Max/Min stays on the tla type.
static Type aveVectorElementType(Type elementType)
{
    if (auto intType = dyn_cast<IntegerType>(elementType))
        if (intType.isUnsigned())
            return IntegerType::get(elementType.getContext(), intType.getWidth());
    return elementType;
}

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
        lanes, elementType, VectorType::get({lanes}, aveVectorElementType(elementType)),
        fullPregVecType(elementType.getContext())};
}

// Return the value already mapped into the helper, or clone an arith.constant
// on demand (loop bounds / index math constants are pulled in lazily this way).
static Value lookupOrCloneScalarValue(OpBuilder& b, Value value, DenseMap<Value, Value>& valueMap)
{
    if (Value mapped = valueMap.lookup(value))
        return mapped;
    if (auto ptrToInt = value.getDefiningOp<::tla::PtrToIntOp>()) {
        if (Value mapped = lookupOrCloneScalarValue(b, ptrToInt.getPtr(), valueMap)) {
            if (mapped.getType().isInteger(64)) {
                valueMap[value] = mapped;
                return mapped;
            }
            if (isa<LLVM::LLVMPointerType>(mapped.getType())) {
                Value asI64 = b.create<LLVM::PtrToIntOp>(ptrToInt.getLoc(), b.getI64Type(), mapped);
                valueMap[value] = asI64;
                return asI64;
            }
        }
    }
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

/// Map ``newPtr`` (as i64) onto the TLA next_ptr SSA and its ``tla.ptrtoint``.
static void mapUpdateNextPtrI64(Value tlaNextPtr, Value asI64, DenseMap<Value, Value>& valueMap)
{
    valueMap[tlaNextPtr] = asI64;
    for (Operation* user : tlaNextPtr.getUsers()) {
        if (auto ptrToInt = dyn_cast<::tla::PtrToIntOp>(user))
            valueMap[ptrToInt.getResult()] = asI64;
    }
}

/// Lower a ``!tla.ptr`` SCF carrier to helper ``i64`` (or nullptr on failure).
///
/// ``valueMap`` may still hold an abstract ``!tla.ptr`` for a TensorDesc /
/// make_tensor base that UPDATE has not yet rewritten via
/// ``mapUpdateNextPtrI64``. Chase ``tla.inttoptr`` / mapped llvm.ptr / i64
/// instead of treating that as a hard error (master used to keep going with
/// the abstract type until a later silent failure).
static Value materializeTlaPtrCarrierAsI64(OpBuilder& b, Location loc, Value tlaPtr, DenseMap<Value, Value>& valueMap)
{
    Type i64Ty = b.getI64Type();
    auto asI64 = [&](Value v) -> Value {
        if (!v)
            return nullptr;
        if (v.getType().isInteger(64))
            return v;
        if (isa<IndexType>(v.getType()))
            return b.create<arith::IndexCastOp>(loc, i64Ty, v).getResult();
        if (isa<LLVM::LLVMPointerType>(v.getType()))
            return b.create<LLVM::PtrToIntOp>(loc, i64Ty, v).getResult();
        return nullptr;
    };

    if (Value mapped = valueMap.lookup(tlaPtr)) {
        if (Value ok = asI64(mapped))
            return ok;
        // Still ``!tla.ptr`` (or other abstract): fall through to defining op.
    }

    if (auto intToPtr = tlaPtr.getDefiningOp<::tla::IntToPtrOp>()) {
        Value addr = lookupOrCloneScalarValue(b, intToPtr.getAddr(), valueMap);
        if (Value ok = asI64(addr)) {
            valueMap[tlaPtr] = ok;
            return ok;
        }
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
        return b.createOrFold<arith::MulIOp>(loc, origin0, origin1);
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
    // Match the AVE (signless) ABI; ctx.elementType may still be unsigned.
    auto castScalar = castScalarForVectorElement(scalar, aveVectorElementType(ctx.elementType));
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
            return b.create<hivmave::VFAddsOp>(loc, ctx.vecType, lhs, scalar, mask).getResult();
        if (info.kind == VectorBinaryKind::Mul)
            return b.create<hivmave::VFMulsOp>(loc, ctx.vecType, lhs, scalar, mask).getResult();
        if (info.kind == VectorBinaryKind::Max) {
            if (isUnsignedInt(ctx.elementType))
                return b.create<hivmave::VMaxsUIOp>(loc, ctx.vecType, lhs, scalar, mask).getResult();
            return b.create<hivmave::VFMaxsOp>(loc, ctx.vecType, lhs, scalar, mask).getResult();
        }
        if (isUnsignedInt(ctx.elementType))
            return b.create<hivmave::VMinsUIOp>(loc, ctx.vecType, lhs, scalar, mask).getResult();
        return b.create<hivmave::VFMinsOp>(loc, ctx.vecType, lhs, scalar, mask).getResult();
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

    // Ptr-backed windows share one effective-address model with UPDATE:
    // prefer the live address (``inttoptr`` or mapped UPDATE next_ptr i64),
    // then apply descriptor coords. ``valueMap`` may hold the storage-identity
    // seed helper arg (for ABI dedup / memref type only) — that seed must not
    // replace a mapped live address. If the live addr is not reachable in the
    // helper, keep the seed memref (ordinary lit / non-UPDATE paths).
    auto rebuildFromAddrI64 = [&](Value addr) {
        Value addrI64 = ::tla::castValueToI64(b, loc, addr);
        SmallVector<Value, 1> dynamicSizes;
        if (!sourceType.hasStaticShape()) {
            for (int64_t dim : sourceType.getShape()) {
                if (dim == ShapedType::kDynamic)
                    dynamicSizes.push_back(b.create<arith::ConstantIndexOp>(loc, 0));
            }
        }
        baseMemref = b.create<hivm::PointerCastOp>(loc, sourceType, addrI64, dynamicSizes).getResult();
    };
    Value descBase = descOp.getBase();
    if (Value mappedBase = valueMap.lookup(descBase)) {
        if (mappedBase.getType().isInteger(64) || isa<IndexType>(mappedBase.getType()))
            rebuildFromAddrI64(mappedBase);
        else if (isa<LLVM::LLVMPointerType>(mappedBase.getType())) {
            rebuildFromAddrI64(b.create<LLVM::PtrToIntOp>(loc, b.getI64Type(), mappedBase));
        }
    } else if (auto intToPtr = descBase.getDefiningOp<::tla::IntToPtrOp>()) {
        if (Value addr = lookupOrCloneScalarValue(b, intToPtr.getAddr(), valueMap))
            rebuildFromAddrI64(addr);
    }

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

/// Effective UB pointer for a tile: same address ordinary load/store and
/// UPDATE `.post` use. Prefer the live ``tensor_desc(inttoptr(i64))`` address
/// so UPDATE next windows that share a storage-identity seed memref still
/// advance. Fall back to the mapped memref view when no live ptr is reachable.
static FailureOr<Value> resolveEffectiveUbPtr(
    Value tensor, OpBuilder& b, Location loc, DenseMap<Value, Value>& valueMap)
{
    if (auto desc = tensor.getDefiningOp<::tla::TensorDescOp>()) {
        Value base = desc.getBase();
        if (Value mappedBase = valueMap.lookup(base)) {
            if (isa<LLVM::LLVMPointerType>(mappedBase.getType()))
                return mappedBase;
            if (mappedBase.getType().isInteger(64))
                return intToUbPtr(b, loc, mappedBase);
        }
        if (auto intToPtr = base.getDefiningOp<::tla::IntToPtrOp>()) {
            Value addr = lookupOrCloneScalarValue(b, intToPtr.getAddr(), valueMap);
            if (addr) {
                Value ptr = intToUbPtr(b, loc, addr);
                valueMap[intToPtr.getResult()] = ptr;
                return ptr;
            }
        }
    }
    if (Value mapped = valueMap.lookup(tensor)) {
        if (isa<MemRefType>(mapped.getType())) {
            auto meta = b.create<mlir::memref::ExtractStridedMetadataOp>(loc, mapped);
            Value basePtr = b.create<mlir::memref::ExtractAlignedPointerAsIndexOp>(loc, meta.getBaseBuffer());
            Value elemOffset = meta.getOffset();
            auto mt = cast<MemRefType>(mapped.getType());
            int64_t elemBytes = getByteSizeOfFixedWidthScalarType(mt.getElementType());
            Value byteOffset = elemOffset;
            if (elemBytes > 1) {
                Value scale = b.create<arith::ConstantIndexOp>(loc, elemBytes);
                byteOffset = b.create<arith::MulIOp>(loc, elemOffset, scale);
            }
            Value byteAddr = b.create<arith::AddIOp>(loc, basePtr, byteOffset);
            return intToUbPtr(b, loc, ::tla::castValueToI64(b, loc, byteAddr));
        }
        if (isa<LLVM::LLVMPointerType>(mapped.getType()))
            return mapped;
    }
    return failure();
}

// Re-create one vec.func body op inside the helper: tla ops become AVE vector
// ops; scf control flow and index arithmetic are carried verbatim. Each op
// derives its own vector/mask width from its operands or result element type,
// so a single region may mix element widths (e.g. across tla.cast).
static LogicalResult lowerLoad(
    ::tla::LoadOp loadOp, OpBuilder& b, ModuleOp module, DenseMap<Value, Value>& valueMap,
    AutoFullPredicateCache& fullMasks)
{
    Operation& op = *loadOp.getOperation();
    Location loc = op.getLoc();
    Value source = valueMap.lookup(loadOp.getSource());
    if (!source)
        return failure();

    // MaskSSA result: 1/2/4-byte UB → predicate.
    // DIST_US / DIST_DS: AscendC MaskDist via linked bc (plds dist=1/2).
    // IR i1→plds hardcodes dist=0, so US/DS must not go through VFLoad.
    if (auto maskType = dyn_cast<::tla::MaskSSAType>(loadOp.getResult().getType())) {
        int64_t lanes = maskType.getPhysicalLanes();
        hivmave::LoadDist pattern = hivmave::LoadDist::NORM;
        if (auto loadDistAttr = loadOp.getLoadDist())
            pattern = mapTlaLoadDistToAve(loadDistAttr->getLoadDist());
        if (pattern != hivmave::LoadDist::NORM && pattern != hivmave::LoadDist::US && pattern != hivmave::LoadDist::DS)
            return loadOp.emitError("mask load_dist must be norm|us|ds"), failure();

        if (pattern == hivmave::LoadDist::US || pattern == hivmave::LoadDist::DS) {
            ModuleOp module = loadOp->getParentOfType<ModuleOp>();
            auto callee = getOrCreateMaskLoadUsDsLibraryCall(module, loc, source.getType(), pattern);
            if (!callee)
                return loadOp.emitError(
                           "MaskLoadDist US/DS requires a b8/b16/b32/b64 "
                           "scalar UB (AscendC LoadAlign<T, MaskDist>)"),
                       failure();
            auto call = b.create<func::CallOp>(loc, callee, ValueRange{source});
            Value loaded = castMaskToPregType(b, loc, call.getResult(0), fullPregVecType(b.getContext()));
            (void)lanes;
            valueMap[loadOp.getResult()] = loaded;
            return success();
        }

        auto i1MemrefOr = materializeI1MaskMemrefFromUb(b, loc, source, lanes);
        if (failed(i1MemrefOr))
            return loadOp.emitError("failed to materialize i1 memref view for tla.load MaskSSA"), failure();
        VectorType semanticMaskType = VectorType::get({lanes}, b.getI1Type());
        Value zero = b.create<arith::ConstantIndexOp>(loc, 0);
        auto vfLoad = createVFLoad(b, loc, semanticMaskType, *i1MemrefOr, zero, pattern, /*unaligned=*/false);
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
    // Dual-dest load_dist needs a second vector result.
    Value r2 = loadOp.getResult2();
    bool hasSecondVec = static_cast<bool>(r2);
    bool dual = isDualDestLoadDist(pattern);
    if (dual != hasSecondVec)
        return loadOp.emitError(
                   "dintlv load_dist requires exactly two vector results; other "
                   "load_dist values require one vector result "
                   "(optional live !tla.ptr is separate)"),
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
        auto callee = getOrCreateLoadWithStrideLibraryCall(module, loc, resultVecType, source.getType(), calleeName);
        if (!callee)
            return failure();
        // The gather reads all 8 DataBlocks, so the predicate is all-true;
        // build it via PgePattern::ALL like the other no-mask lowerings
        // (a dense<true> arith constant is not selectable in bisheng).
        Value allTrue = allTrueMaskFor(fullMasks, loc, opCtx->vecType, loadOp.getResult().getType());
        auto call = b.create<func::CallOp>(loc, callee, ValueRange{source, blockStrideVal, repeatStrideVal, allTrue});
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
        valueMap[r2] = vfLoad.getRes1();
    return success();
}

static LogicalResult materializeBlockStrides(
    Operation* op, Value blockStride, Value repeatStride, OpBuilder& b, DenseMap<Value, Value>& valueMap,
    Value& blockOut, Value& repeatOut)
{
    Location loc = op->getLoc();
    blockOut = valueMap.lookup(blockStride);
    if (!blockOut)
        blockOut = lookupOrCloneScalarValue(b, blockStride, valueMap);
    repeatOut = valueMap.lookup(repeatStride);
    if (!repeatOut)
        repeatOut = lookupOrCloneScalarValue(b, repeatStride, valueMap);
    if (!blockOut || !repeatOut)
        return op->emitError("failed to materialize DATA_BLOCK_COPY strides"), failure();
    blockOut = castToI32(b, loc, blockOut);
    repeatOut = castToI32(b, loc, repeatOut);
    return success();
}

static LogicalResult lowerLoadUpdate(
    ::tla::LoadUpdateOp loadOp, OpBuilder& b, ModuleOp module, DenseMap<Value, Value>& valueMap,
    AutoFullPredicateCache& fullMasks)
{
    Location loc = loadOp.getLoc();
    Value source = valueMap.lookup(loadOp.getSource());
    if (!source)
        return loadOp.emitError("load_update source is not mapped in vector helper"), failure();
    auto sourceType = dyn_cast<MemRefType>(source.getType());
    if (!sourceType)
        return loadOp.emitError("load_update source map is not a memref in vector helper"), failure();
    auto opCtx = deriveVecCtxForElement(sourceType.getElementType());
    if (failed(opCtx))
        return loadOp.emitError("unsupported element type for load_update vector helper"), failure();
    auto ptrOr = resolveEffectiveUbPtr(loadOp.getSource(), b, loc, valueMap);
    if (failed(ptrOr))
        return loadOp.emitError(
                   "load_update requires an effective UB address "
                   "(tensor_desc over live inttoptr / materialized memref)"),
               failure();

    // DATA_BLOCK_COPY UPDATE: no HIVMC vsldb.post — AIV bc returns the pointer.
    if (Value blockStrideVal = loadOp.getBlockStride()) {
        Value blockStride;
        Value repeatStride;
        if (failed(materializeBlockStrides(
                loadOp, blockStrideVal, loadOp.getPostUpdateStride(), b, valueMap, blockStride, repeatStride)))
            return failure();
        std::string suffix = dataBlockCopyElementSuffix(sourceType.getElementType());
        if (suffix.empty())
            return loadOp.emitError("unsupported element type for DATA_BLOCK_COPY load_update: ")
                       << sourceType.getElementType(),
                   failure();
        bool viaBytes = blockStrideUsesByteRegister(sourceType.getElementType());
        Type resultVecType = viaBytes ? byteRegisterTypeFor(opCtx->vecType) : Type(opCtx->vecType);
        auto callee = getOrCreateLoadUpdateRetptrLibraryCall(
            module, loc, resultVecType, ("_mlir_ciface_load_with_stride_post_update_retptr_" + suffix));
        Value allTrue = allTrueMaskFor(fullMasks, loc, opCtx->vecType, loadOp.getResult().getType());
        Value preg = castMaskToPregType(b, loc, allTrue, fullPregVecType(b.getContext()));
        auto slotOr = fullMasks.createI64RetptrSlot(b, loc);
        if (failed(slotOr))
            return loadOp.emitError("failed to allocate load_update retptr slot in outlined func entry"), failure();
        auto call = b.create<func::CallOp>(loc, callee, ValueRange{*ptrOr, blockStride, repeatStride, preg, *slotOr});
        Value loaded = call.getResult(0);
        if (viaBytes)
            loaded = b.create<mlir::vector::BitCastOp>(loc, opCtx->vecType, loaded).getResult();
        Value newAddr = b.create<LLVM::LoadOp>(loc, b.getI64Type(), *slotOr);
        valueMap[loadOp.getResult()] = loaded;
        mapUpdateNextPtrI64(loadOp.getNextPtr(), newAddr, valueMap);
        return success();
    }

    Value stride = valueMap.lookup(loadOp.getPostUpdateStride());
    if (!stride)
        stride = lookupOrCloneScalarValue(b, loadOp.getPostUpdateStride(), valueMap);
    if (!stride)
        return loadOp.emitError("failed to materialize post_update_stride in vector helper"), failure();
    Type elemTy = sourceType.getElementType();
    FailureOr<std::pair<Value, Value>> emitted = emitVldsx1Post(b, loc, *ptrOr, stride, elemTy, opCtx->vecType);
    if (failed(emitted))
        return loadOp.emitError("unsupported element type for tla.load_update: ") << elemTy, failure();
    valueMap[loadOp.getResult()] = emitted->first;
    mapUpdateNextPtrI64(
        loadOp.getNextPtr(), b.create<LLVM::PtrToIntOp>(loc, b.getI64Type(), emitted->second), valueMap);
    return success();
}

static LogicalResult lowerLoadUnalignUpdate(
    ::tla::LoadUnalignUpdateOp op, OpBuilder& b, DenseMap<Value, Value>& valueMap)
{
    Location loc = op.getLoc();
    // AscendC: HIVM $cceVL = postUpdateStride×sizeof(T); next_ptr = HW newPtr.
    Value postUpdateStride = valueMap.lookup(op.getPostUpdateStride());
    if (!postUpdateStride)
        postUpdateStride = lookupOrCloneScalarValue(b, op.getPostUpdateStride(), valueMap);
    if (!postUpdateStride)
        return op.emitError("failed to materialize post_update_stride"), failure();
    auto ptrOr = resolveEffectiveUbPtr(op.getSource(), b, loc, valueMap);
    if (failed(ptrOr))
        return op.emitError("load_unalign_update requires a live UB address"), failure();
    Value source = valueMap.lookup(op.getSource());
    auto sourceType = dyn_cast_or_null<MemRefType>(source ? source.getType() : Type());
    if (!sourceType)
        return failure();
    Type elemType = sourceType.getElementType();
    Value countBytes = strideElemsToByteOffI32(b, loc, postUpdateStride, elemType);
    // Dual instr: local vldas + vldus.post (self-contained; Pre not carried).
    FailureOr<Value> localUreg = emitVldas(b, loc, *ptrOr);
    if (failed(localUreg))
        return op.emitError("failed to emit vldas for unalign UPDATE"), failure();
    FailureOr<std::tuple<Value, Value, Value>> emitted =
        emitVldusPost(b, loc, *ptrOr, *localUreg, countBytes, elemType);
    if (failed(emitted))
        return op.emitError("unsupported element type for tla.load_unalign_update"), failure();
    valueMap[op.getResult()] = std::get<0>(*emitted);
    mapUpdateNextPtrI64(
        op.getNextPtr(), b.create<LLVM::PtrToIntOp>(loc, b.getI64Type(), std::get<2>(*emitted)), valueMap);
    return success();
}

/// tla.bitcast: same-width bit reinterpretation. The register keeps its bits
/// and only the element type changes, so this lowers to MLIR's vector.bitcast.
static LogicalResult lowerBitCast(::tla::BitCastOp bitCastOp, OpBuilder& b, DenseMap<Value, Value>& valueMap)
{
    Location loc = bitCastOp.getLoc();
    Value src = valueMap.lookup(bitCastOp.getSource());
    if (!src)
        return failure();
    auto srcVecType = dyn_cast<VectorType>(src.getType());
    if (!srcVecType)
        return bitCastOp.emitError("tla.bitcast source is not a vector value"), failure();
    Type dstElementType = bitCastOp.getResult().getType().getElementType();
    // Values stay signless; signedness lives in the !tla.vector type only.
    if (auto intType = dyn_cast<IntegerType>(dstElementType))
        if (!intType.isSignless())
            dstElementType = b.getIntegerType(intType.getWidth());
    auto dstVecType = VectorType::get(srcVecType.getShape(), dstElementType);
    valueMap[bitCastOp.getResult()] = b.create<mlir::vector::BitCastOp>(loc, dstVecType, src).getResult();
    return success();
}

/// tla.cast: element-type conversion. The source vector already carries its
/// width; the destination width is one full 256-byte register's worth of the
/// target element type. The cast op picks the AVE cast (vtruncf / vfptosi /
/// vsitofp / vtrunci / ...) from the (src,dst) element kinds.
static LogicalResult lowerCast(
    ::tla::CastOp castOp, OpBuilder& b, ModuleOp module, DenseMap<Value, Value>& valueMap,
    AutoFullPredicateCache& fullMasks)
{
    Location loc = castOp.getLoc();
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
    Type tlaSrcElem = cast<::tla::VectorSSAType>(castOp.getSource().getType()).getElementType();
    Type tlaDstElem = dstType.getElementType();
    auto dstVecType = VectorType::get({*dstLanesOr}, aveVectorElementType(tlaDstElem));
    // Reject casts whose source or destination element type has no AVE cast path
    // (i1/bool, f64). Unsigned routes are gated by createUnsignedVectorCastResult.
    if (!isSupportedCastElementType(tlaSrcElem) || !isSupportedCastElementType(tlaDstElem))
        return castOp.emitError(
                   "unsupported tla.cast element type: only signed "
                   "integers (i8/i16/i32/i64), unsigned integers "
                   "(u8/u16/u32) and floats (f16/bf16/f32) "
                   "are supported; bool and f64 are not"),
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
    auto result = createVectorCastResult(
        b, fullMasks, loc, srcVecType, dstVecType, tlaSrcElem, tlaDstElem, trait, src, mask, (bool)castOp.getMask());
    if (failed(result))
        return castOp.emitError("unsupported tla.cast element type conversion"), failure();
    valueMap[castOp.getResult()] = *result;
    return success();
}

static LogicalResult lowerFull(
    ::tla::FullOp fullOp, OpBuilder& b, ModuleOp module, DenseMap<Value, Value>& valueMap,
    AutoFullPredicateCache& fullMasks)
{
    Operation& op = *fullOp.getOperation();
    Location loc = op.getLoc();
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

static LogicalResult lowerArange(
    ::tla::ArangeOp arangeOp, OpBuilder& b, ModuleOp module, DenseMap<Value, Value>& valueMap,
    AutoFullPredicateCache& fullMasks)
{
    Location loc = arangeOp.getLoc();
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

static LogicalResult lowerVectorBinary(
    Operation& op, const VectorOpInfo* info, OpBuilder& b, ModuleOp module, DenseMap<Value, Value>& valueMap,
    AutoFullPredicateCache& fullMasks)
{
    Location loc = op.getLoc();
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
    Value mask;
    if (operands.mask) {
        mask = valueMap.lookup(operands.mask);
        if (!mask)
            return failure();
    } else {
        mask = allTrueMaskFor(fullMasks, loc, opVecType, operands.lhs.getType());
    }
    // createVectorBinaryResult peels unsigned from the tla lhs type.
    Value result = createVectorBinaryResult(
        b, loc, info->kind, operands.lhs.getType(), opVecType.getElementType(), opVecType, lhs, rhs, mask);
    if (!result)
        return failure();
    valueMap[op.getResult(0)] = result;
    return success();
}

static LogicalResult lowerVectorScalarBinary(
    Operation& op, const VectorOpInfo* info, OpBuilder& b, ModuleOp module, DenseMap<Value, Value>& valueMap,
    AutoFullPredicateCache& fullMasks)
{
    Location loc = op.getLoc();
    if (op.getNumResults() != 1)
        return failure();
    TlaBinaryOperands operands = info->operands;
    Value lhs = valueMap.lookup(operands.lhs);
    if (!lhs)
        return failure();
    // Derive ctx from tla element so unsigned max/min still select VMaxsUI.
    auto lhsTy = dyn_cast<VectorType>(lhs.getType());
    if (!lhsTy)
        return failure();
    auto opCtx = deriveVecCtxForElement(tlaElementOr(operands.lhs.getType(), lhsTy.getElementType()));
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

/// tla.squeeze: mask-compress src lanes via linked bitcode (vsqz). Uses
/// NO_STORE_REG; STORE_REG + StoreUnAlign streaming writeback is not exposed
/// until unaligned store (StoreUnAlign/StoreUnAlignPost) is available in TLA.
static LogicalResult lowerSqueeze(
    ::tla::SqueezeOp squeezeOp, OpBuilder& b, ModuleOp module, DenseMap<Value, Value>& valueMap,
    AutoFullPredicateCache& fullMasks)
{
    Location loc = squeezeOp.getLoc();
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
        return squeezeOp.emitError("unsupported element type for tla.squeeze: ") << srcTy.getElementType(), failure();
    VectorType pregVecType = fullPregVecType(b.getContext());
    Value preg = castMaskToPregType(b, loc, mask, pregVecType);
    auto callee = getOrCreateSqueezeLibraryCall(module, loc, opCtx->vecType, pregVecType, calleeName);
    Value result = b.create<func::CallOp>(loc, callee, ValueRange{src, preg}).getResult(0);
    valueMap[squeezeOp.getResult()] = result;
    return success();
}

static LogicalResult lowerReduce(
    ::tla::ReduceOp reduceOp, OpBuilder& b, ModuleOp module, DenseMap<Value, Value>& valueMap,
    AutoFullPredicateCache& fullMasks)
{
    Operation& op = *reduceOp.getOperation();
    Location loc = op.getLoc();
    if (op.getNumResults() != 1)
        return failure();
    Value operand = valueMap.lookup(reduceOp->getOperand(0));
    if (!operand)
        return failure();
    // AVE operand is signless; CombiningKind UMAX/UMIN needs the tla element type.
    auto operandTy = dyn_cast<VectorType>(operand.getType());
    if (!operandTy)
        return failure();
    Type tlaElem = tlaElementOr(reduceOp->getOperand(0).getType(), operandTy.getElementType());
    auto opCtx = deriveVecCtxForElement(tlaElem);
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

// tla.reduce_intra_block: per-32B-block group reduction
// lowered directly to the Vcg regbase instructions. The container value is
// already a full 256-byte register vector, so no pad is required: the
// instruction treats inactive (mask=0) elements as identity values (0 for ADD,
// -INF/+INF for MAX/MIN) and packs each block's result contiguously into the
// LSBs of the destination. The user mask is passed through unchanged.
static LogicalResult lowerReduceIntraBlock(
    ::tla::ReduceIntraBlockOp rdbOp, OpBuilder& b, ModuleOp module, DenseMap<Value, Value>& valueMap,
    AutoFullPredicateCache& fullMasks)
{
    Operation& op = *rdbOp.getOperation();
    Location loc = op.getLoc();
    Value operand = valueMap.lookup(rdbOp.getOperand());
    if (!operand)
        return failure();
    auto operandTy = dyn_cast<VectorType>(operand.getType());
    if (!operandTy)
        return failure();
    // The Vcg S/U variant selection follows the tla element type; the AVE
    // container element is signless.
    Type tlaElem = tlaElementOr(rdbOp->getOperand(0).getType(), operandTy.getElementType());
    auto opCtx = deriveVecCtxForElement(tlaElem);
    if (failed(opCtx))
        return failure();
    if (!rdbOp.getMask())
        return rdbOp.emitError("tla.reduce_intra_block requires an explicit mask"), failure();
    Value mask = valueMap.lookup(rdbOp.getMask());
    if (!mask)
        return failure();
    VectorType pregVecType = fullPregVecType(b.getContext());
    Value preg = castMaskToPregType(b, loc, mask, pregVecType);

    auto kindAttr = rdbOp->getAttrOfType<StringAttr>("kind");
    if (!kindAttr)
        return rdbOp.emitError("tla.reduce_intra_block requires string kind attribute"), failure();
    StringRef kind = kindAttr.getValue();
    // Defensive mirror of the ReduceIntraBlockOp verifier constraints: the
    // Vcg dst is presented as the op's TLA result type, so a hand-written op
    // whose result type violates the widening/mirror rules would lower into
    // mismatched IR. Reject it here instead of ignoring the result type.
    auto tlaOperandTy = dyn_cast<::tla::VectorSSAType>(rdbOp.getOperand().getType());
    auto tlaResultTy = dyn_cast<::tla::VectorSSAType>(rdbOp.getResult().getType());
    if (!tlaResultTy)
        return rdbOp.emitError("tla.reduce_intra_block result must be !tla.vector"), failure();
    auto opElem = dyn_cast<IntegerType>(tlaElem);
    auto resElem = dyn_cast<IntegerType>(tlaResultTy.getElementType());
    bool widenAdd = (kind == "add") && opElem && (opElem.getWidth() == 16);
    if (widenAdd) {
        // The widening keeps the operand's sign category: i16 -> i32, ui16 -> ui32.
        if (!resElem || (resElem.getWidth() != 32) || (resElem.getSignedness() != opElem.getSignedness()))
            return rdbOp.emitError() << "add on i16/ui16 widens the result to the matching 32-bit type "
                                     << "(i16->i32, ui16->ui32), got " << tlaResultTy.getElementType(),
                   failure();
        // Lane count halves per 32-bit result lane, rounding up so a tail's
        // last valid element keeps its widened lane.
        if ((tlaOperandTy.getValidLanes() != ShapedType::kDynamic) &&
            (tlaResultTy.getValidLanes() != ((tlaOperandTy.getValidLanes() + 1) / 2)))
            return rdbOp.emitError() << "add on i16/ui16 keeps the 256B container, expected "
                                     << ((tlaOperandTy.getValidLanes() + 1) / 2) << " result lane(s), got "
                                     << tlaResultTy.getValidLanes(),
                   failure();
    } else if (
        (tlaResultTy.getElementType() != tlaElem) ||
        (tlaOperandTy && (tlaResultTy.getValidLanes() != tlaOperandTy.getValidLanes()))) {
        return rdbOp.emitError("result type must match operand type (packed mirror)"), failure();
    }
    bool isUnsigned = false;
    if (auto intType = dyn_cast<IntegerType>(tlaElem))
        isUnsigned = intType.isUnsigned();

    Value result;
    if (kind == "add") {
        // ADD on 16-bit sources accumulates into a 32-bit destination (the same
        // widening behavior as vcadd): each block sum is written as a 32-bit
        // value into the LSBs. The LLVM-side intrinsic signature requires the
        // source to have the destination's type, so the Vcgadd is emitted on the
        // 16-bit container and the result register is re-presented as a 64-lane
        // i32 vector with a zero-cost bitcast. No Vcvtii cast back: block sums
        // beyond the source type's range stay exact in the i32/ui32 result, and
        // truncation is the user's decision.
        result = isUnsigned ?
                     b.create<hivm_regbaseintrins::VcgaddUXInstrOp>(loc, opCtx->vecType, operand, preg).getRes() :
                     b.create<hivm_regbaseintrins::VcgaddSXInstrOp>(loc, opCtx->vecType, operand, preg).getRes();
        if (isa<IntegerType>(tlaElem) && (dyn_cast<IntegerType>(tlaElem).getWidth() == 16)) {
            VectorType wideType = VectorType::get({64}, b.getI32Type());
            result = b.create<mlir::vector::BitCastOp>(loc, wideType, result).getResult();
        }
    } else if (kind == "max") {
        result = isUnsigned ?
                     b.create<hivm_regbaseintrins::VcgmaxUInstrOp>(loc, opCtx->vecType, operand, preg).getRes() :
                     b.create<hivm_regbaseintrins::VcgmaxSInstrOp>(loc, opCtx->vecType, operand, preg).getRes();
    } else if (kind == "min") {
        result = isUnsigned ?
                     b.create<hivm_regbaseintrins::VcgminUInstrOp>(loc, opCtx->vecType, operand, preg).getRes() :
                     b.create<hivm_regbaseintrins::VcgminSInstrOp>(loc, opCtx->vecType, operand, preg).getRes();
    } else {
        return rdbOp.emitError() << "tla.reduce_intra_block supports only add, max, and min reductions, got \"" << kind
                                 << "\"",
               failure();
    }
    valueMap[op.getResult(0)] = result;
    return success();
}

static LogicalResult lowerPack(::tla::PackOp packOp, OpBuilder& b, ModuleOp module, DenseMap<Value, Value>& valueMap)
{
    Location loc = packOp.getLoc();
    Value src = valueMap.lookup(packOp.getSrc());
    if (!src)
        return failure();

    if (isa<::tla::VectorSSAType>(packOp.getSrc().getType())) {
        auto srcType = cast<::tla::VectorSSAType>(packOp.getSrc().getType());
        auto dstType = cast<::tla::VectorSSAType>(packOp.getResult().getType());
        auto srcCtx = deriveVecCtxForElement(srcType.getElementType());
        auto dstCtx = deriveVecCtxForElement(dstType.getElementType());
        if (failed(srcCtx) || failed(dstCtx))
            return failure();
        auto name = getVectorPackLibraryCallName(srcType.getElementType(), dstType.getElementType(), packOp.getPart());
        auto callee = getOrCreateVectorPackUnpackLibraryCall(module, loc, srcCtx->vecType, dstCtx->vecType, name);
        valueMap[packOp.getResult()] = b.create<func::CallOp>(loc, callee, ValueRange{src}).getResult(0);
        return success();
    }
    if (isa<::tla::MaskSSAType>(packOp.getSrc().getType())) {
        int64_t partValue = packOp.getPart() == ::RegPart::high ? 1 : 0;
        Value part = b.create<arith::ConstantIntOp>(loc, partValue, 32);
        VectorType pregType = fullPregVecType(b.getContext());
        Value pregSrc = castMaskToPregType(b, loc, src, pregType);
        auto packed = b.create<hivm_regbaseintrins::PpackInstrOp>(loc, pregType, pregSrc, part);
        valueMap[packOp.getResult()] = packed->getResult(0);
        return success();
    }
    return failure();
}

static LogicalResult lowerUnpack(
    ::tla::UnpackOp unpackOp, OpBuilder& b, ModuleOp module, DenseMap<Value, Value>& valueMap)
{
    Value src = valueMap.lookup(unpackOp.getSrc());
    if (!src)
        return failure();
    // Preserve TLA signedness in the symbol; CANN selects sign/zero extension.
    auto srcType = cast<::tla::VectorSSAType>(unpackOp.getSrc().getType());
    auto dstType = cast<::tla::VectorSSAType>(unpackOp.getResult().getType());
    auto srcCtx = deriveVecCtxForElement(srcType.getElementType());
    auto dstCtx = deriveVecCtxForElement(dstType.getElementType());
    if (failed(srcCtx) || failed(dstCtx))
        return failure();
    auto name = getVectorUnpackLibraryCallName(srcType.getElementType(), dstType.getElementType(), unpackOp.getPart());
    auto callee =
        getOrCreateVectorPackUnpackLibraryCall(module, unpackOp.getLoc(), srcCtx->vecType, dstCtx->vecType, name);
    valueMap[unpackOp.getResult()] = b.create<func::CallOp>(unpackOp.getLoc(), callee, ValueRange{src}).getResult(0);
    return success();
}

static LogicalResult lowerInterleave(
    ::tla::InterleaveOp interleaveOp, OpBuilder& b, ModuleOp module, DenseMap<Value, Value>& valueMap,
    AutoFullPredicateCache& fullMasks)
{
    Operation& op = *interleaveOp.getOperation();
    Location loc = op.getLoc();
    if (op.getNumResults() != 2)
        return failure();

    Value src0 = valueMap.lookup(interleaveOp.getSrc0());
    Value src1 = valueMap.lookup(interleaveOp.getSrc1());
    if (!src0 || !src1)
        return failure();

    if (isa<::tla::MaskSSAType>(interleaveOp.getSrc0().getType())) {
        auto maskType = cast<::tla::MaskSSAType>(interleaveOp.getSrc0().getType());
        auto pregType = fullPregVecType(b.getContext());
        src0 = castMaskToPregType(b, loc, src0, pregType);
        src1 = castMaskToPregType(b, loc, src1, pregType);
        auto pairType = LLVM::LLVMStructType::getLiteral(b.getContext(), {pregType, pregType});
        Operation* intrinsic = nullptr;
        switch (maskType.getPhysicalLanes()) {
            case 256:
                intrinsic = b.create<hivm_regbaseintrins::PintlvB8InstrOp>(loc, pairType, src0, src1);
                break;
            case 128:
                intrinsic = b.create<hivm_regbaseintrins::PintlvB16InstrOp>(loc, pairType, src0, src1);
                break;
            case 64:
                intrinsic = b.create<hivm_regbaseintrins::PintlvB32InstrOp>(loc, pairType, src0, src1);
                break;
            default:
                return interleaveOp.emitError("supports only b8, b16, and b32 predicate interleave"), failure();
        }
        Value aggregate = intrinsic->getResult(0);
        valueMap[interleaveOp.getDst0()] = b.create<LLVM::ExtractValueOp>(loc, aggregate, 0);
        valueMap[interleaveOp.getDst1()] = b.create<LLVM::ExtractValueOp>(loc, aggregate, 1);
        return success();
    }

    auto src0Type = dyn_cast<VectorType>(src0.getType());
    auto src1Type = dyn_cast<VectorType>(src1.getType());
    if (!src0Type || !src1Type || src0Type != src1Type)
        return failure();

    auto aveOp = b.create<hivmave::VFInterleaveOp>(loc, TypeRange{src0Type, src1Type}, ValueRange{src0, src1});

    valueMap[op.getResult(0)] = aveOp->getResult(0);
    valueMap[op.getResult(1)] = aveOp->getResult(1);
    return success();
}

static LogicalResult lowerDeInterleave(
    ::tla::DeInterleaveOp deInterleaveOp, OpBuilder& b, ModuleOp module, DenseMap<Value, Value>& valueMap,
    AutoFullPredicateCache& fullMasks)
{
    Operation& op = *deInterleaveOp.getOperation();
    Location loc = op.getLoc();
    if (op.getNumResults() != 2)
        return failure();

    Value src0 = valueMap.lookup(deInterleaveOp.getSrc0());
    Value src1 = valueMap.lookup(deInterleaveOp.getSrc1());
    if (!src0 || !src1)
        return failure();

    if (isa<::tla::MaskSSAType>(deInterleaveOp.getSrc0().getType())) {
        // SCF carriers are vector<256xi1>; choose PDintlv from the original
        // MaskSSA type rather than the converted carrier.
        auto maskType = cast<::tla::MaskSSAType>(deInterleaveOp.getSrc0().getType());
        auto pregType = fullPregVecType(b.getContext());
        src0 = castMaskToPregType(b, loc, src0, pregType);
        src1 = castMaskToPregType(b, loc, src1, pregType);
        auto pairType = LLVM::LLVMStructType::getLiteral(b.getContext(), {pregType, pregType});
        Operation* intrinsic = nullptr;
        switch (maskType.getPhysicalLanes()) {
            case 256:
                intrinsic = b.create<hivm_regbaseintrins::PDintlvB8InstrOp>(loc, pairType, src0, src1);
                break;
            case 128:
                intrinsic = b.create<hivm_regbaseintrins::PDintlvB16InstrOp>(loc, pairType, src0, src1);
                break;
            case 64:
                intrinsic = b.create<hivm_regbaseintrins::PDintlvB32InstrOp>(loc, pairType, src0, src1);
                break;
            default:
                return deInterleaveOp.emitError("supports only b8, b16, and b32 predicate deinterleave"), failure();
        }
        Value aggregate = intrinsic->getResult(0);
        valueMap[deInterleaveOp.getDst0()] = b.create<LLVM::ExtractValueOp>(loc, aggregate, 0);
        valueMap[deInterleaveOp.getDst1()] = b.create<LLVM::ExtractValueOp>(loc, aggregate, 1);
    } else {
        auto src0Type = dyn_cast<VectorType>(src0.getType());
        auto src1Type = dyn_cast<VectorType>(src1.getType());
        if (!src0Type || !src1Type || src0Type != src1Type)
            return failure();
        auto aveOp = b.create<hivmave::VFDeInterleaveOp>(loc, TypeRange{src0Type, src1Type}, ValueRange{src0, src1});
        valueMap[deInterleaveOp.getDst0()] = aveOp->getResult(0);
        valueMap[deInterleaveOp.getDst1()] = aveOp->getResult(1);
    }
    return success();
}

/// tla.gather: per-lane indexed load from a UB tile.
/// x (tile_view → rank-1 memref) → VFGatherOp base
/// y (loaded index vector)        → index_vec
/// mask (optional)                → mask (all-true if absent)
static LogicalResult lowerGather(
    ::tla::GatherOp gatherOp, OpBuilder& b, ModuleOp module, DenseMap<Value, Value>& valueMap,
    AutoFullPredicateCache& fullMasks)
{
    Location loc = gatherOp.getLoc();
    Value src = valueMap.lookup(gatherOp.getX());
    Value indexVec = valueMap.lookup(gatherOp.getY());
    if (!src || !indexVec)
        return failure();

    // Reg→Reg permute via linked bitcode (vselr). Not HIVMAVE vgather / vsel.
    if (isa<::tla::VectorSSAType>(gatherOp.getX().getType())) {
        auto srcTy = dyn_cast<VectorType>(src.getType());
        if (!srcTy)
            return failure();
        auto opCtx = deriveVecCtxForElement(srcTy.getElementType());
        if (failed(opCtx))
            return failure();
        std::string calleeName = getGatherRegLibraryCallName(srcTy.getElementType());
        if (calleeName.empty())
            return gatherOp.emitError("unsupported element type for Reg→Reg gather: ") << srcTy.getElementType(),
                   failure();

        unsigned indexBits = srcTy.getElementType().getIntOrFloatBitWidth();
        VectorType indexUType = VectorType::get(
            opCtx->vecType.getShape(), IntegerType::get(b.getContext(), indexBits, IntegerType::Unsigned));
        if (indexVec.getType() != indexUType)
            indexVec = b.create<UnrealizedConversionCastOp>(loc, indexUType, indexVec).getResult(0);

        auto callee = getOrCreateGatherRegLibraryCall(module, loc, opCtx->vecType, indexUType, calleeName);
        valueMap[gatherOp.getResult()] = b.create<func::CallOp>(loc, callee, ValueRange{src, indexVec}).getResult(0);
        return success();
    }

    // UB→Reg: per-lane indexed load from a UB tile.
    auto baseType = dyn_cast<MemRefType>(src.getType());
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
        b.create<hivmave::VFGatherOp>(loc, resultVecType, src, ValueRange(zero), indexVec, mask);
    return success();
}

static LogicalResult lowerVectorUnary(
    Operation& op, const VectorUnaryInfo* info, OpBuilder& b, ModuleOp module, DenseMap<Value, Value>& valueMap,
    AutoFullPredicateCache& fullMasks)
{
    Location loc = op.getLoc();
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

/// tla.update_mask: tail mask + remaining count. Lowers to ave.hir.plt,
/// whose mask result drives masked stores and whose second result
/// (true_shape - lanes) is threaded back as the loop-carried tail counter.
/// The op's own dtype attr fixes the lane count (256 bytes / element size)
/// and hence the i1 mask width and the tail decrement.
static LogicalResult lowerUpdateMask(
    ::tla::UpdateMaskOp updateMaskOp, OpBuilder& b, ModuleOp module, DenseMap<Value, Value>& valueMap,
    AutoFullPredicateCache& fullMasks)
{
    Location loc = updateMaskOp.getLoc();
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

static LogicalResult lowerCmp(
    ::tla::CmpOp cmpOp, OpBuilder& b, ModuleOp module, DenseMap<Value, Value>& valueMap,
    AutoFullPredicateCache& fullMasks)
{
    Location loc = cmpOp.getLoc();
    Value lhs = valueMap.lookup(cmpOp.getLhs());
    if (!lhs)
        return failure();
    // The compare's operand width fixes both the input vectors and the i1 mask
    // result width. Prefer the tla element type so unsigned compares keep
    // unsigned semantics after aveVectorElementType maps the AVE vector to
    // signless.
    Type tlaElem = lhs.getType();
    if (auto tlaVec = dyn_cast<::tla::VectorSSAType>(cmpOp.getLhs().getType()))
        tlaElem = tlaVec.getElementType();
    else if (auto lhsTy = dyn_cast<VectorType>(lhs.getType()))
        tlaElem = lhsTy.getElementType();
    auto opCtx = deriveVecCtxForElement(tlaElem);
    if (failed(opCtx))
        return failure();
    auto cmpType = mapCmpMode(cmpOp.getMode());
    if (!cmpType)
        return cmpOp.emitError("unknown tla.cmp mode: ") << cmpOp.getMode(), failure();
    // With signless AVE vectors, HIVMCmpOpLowering's isUnsignedInteger path
    // never fires. Remap relational modes to the explicit unsigned CmpType
    // variants so VCmpU* is selected.
    if (auto intTy = dyn_cast<IntegerType>(tlaElem); intTy && intTy.isUnsigned()) {
        switch (*cmpType) {
            case hivmave::CmpType::LT:
                cmpType = hivmave::CmpType::ULT;
                break;
            case hivmave::CmpType::GT:
                cmpType = hivmave::CmpType::UGT;
                break;
            case hivmave::CmpType::LE:
                cmpType = hivmave::CmpType::ULE;
                break;
            case hivmave::CmpType::GE:
                cmpType = hivmave::CmpType::UGE;
                break;
            default:
                break;
        }
    }
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
        Value scalar = *scalarOr;
        // VFCmpS scalar ODS is signless; keep the vector unsigned for VCmpU*.
        if (auto intTy = dyn_cast<IntegerType>(scalar.getType())) {
            if (intTy.isUnsigned()) {
                Type signless = IntegerType::get(b.getContext(), intTy.getWidth());
                scalar = b.create<UnrealizedConversionCastOp>(loc, signless, scalar).getResult(0);
            }
        }
        valueMap[cmpOp.getResult()] = b.create<hivmave::VFCmpS>(loc, opCtx->maskVecType, *cmpType, lhs, scalar, mask);
    }
    return success();
}

static LogicalResult lowerStore(
    ::tla::StoreOp storeOp, OpBuilder& b, ModuleOp module, DenseMap<Value, Value>& valueMap,
    AutoFullPredicateCache& fullMasks)
{
    Location loc = storeOp.getLoc();
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
    if (storeOp.getUnalignedUbAccess().value_or(false)) {
        // store_unalign (NORMAL): mark AVE masked-store as unaligned UB access.
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

static LogicalResult lowerStoreUpdate(
    ::tla::StoreUpdateOp storeOp, OpBuilder& b, ModuleOp module, DenseMap<Value, Value>& valueMap,
    AutoFullPredicateCache& fullMasks)
{
    Location loc = storeOp.getLoc();
    Value dest = valueMap.lookup(storeOp.getDest());
    Value source = valueMap.lookup(storeOp.getSource());
    if (!dest || !source)
        return failure();
    auto sourceTy = dyn_cast<VectorType>(source.getType());
    if (!sourceTy)
        return failure();
    Value mask;
    auto opCtx = deriveVecCtxForElement(sourceTy.getElementType());
    if (failed(opCtx))
        return failure();
    if (storeOp.getMask()) {
        mask = valueMap.lookup(storeOp.getMask());
        if (!mask)
            return failure();
    } else if (auto destStaticLanes = getStaticTlaTensorValidLaneCount(storeOp.getDest())) {
        int64_t activeLanes = *destStaticLanes >= opCtx->lanes ? opCtx->lanes : *destStaticLanes;
        if (activeLanes == opCtx->lanes)
            mask = fullMasks.get(loc, opCtx->lanes);
        else {
            Value active = b.create<arith::ConstantIndexOp>(loc, activeLanes);
            mask = createPredicatePlt(b, loc, opCtx->maskVecType, opCtx->lanes, active).getRes();
        }
    } else {
        auto destLanes = getTlaTensorValidLaneCount(b, loc, storeOp.getDest(), valueMap);
        if (failed(destLanes))
            return failure();
        mask = createPredicatePlt(b, loc, opCtx->maskVecType, opCtx->lanes, *destLanes).getRes();
    }
    VectorType pregVecType = fullPregVecType(b.getContext());
    Value pregMask = castMaskToPregType(b, loc, mask, pregVecType);
    auto ptrOr = resolveEffectiveUbPtr(storeOp.getDest(), b, loc, valueMap);
    if (failed(ptrOr))
        return storeOp.emitError(
                   "store_update requires an effective UB address "
                   "(tensor_desc over live inttoptr / materialized memref)"),
               failure();

    // DATA_BLOCK_COPY UPDATE: no HIVMC vsstb.post — AIV bc returns the pointer.
    if (Value blockStrideVal = storeOp.getBlockStride()) {
        Value blockStride;
        Value repeatStride;
        if (failed(materializeBlockStrides(
                storeOp, blockStrideVal, storeOp.getPostUpdateStride(), b, valueMap, blockStride, repeatStride)))
            return failure();
        std::string suffix = dataBlockCopyElementSuffix(sourceTy.getElementType());
        if (suffix.empty())
            return storeOp.emitError("unsupported element type for DATA_BLOCK_COPY store_update: ")
                       << sourceTy.getElementType(),
                   failure();
        Value storeSrc = source;
        if (blockStrideUsesByteRegister(sourceTy.getElementType()))
            storeSrc = b.create<mlir::vector::BitCastOp>(loc, byteRegisterTypeFor(sourceTy), source).getResult();
        auto callee = getOrCreateStoreUpdateRetptrLibraryCall(
            module, loc, storeSrc.getType(), ("_mlir_ciface_store_with_stride_post_update_retptr_" + suffix));
        auto call =
            b.create<func::CallOp>(loc, callee, ValueRange{storeSrc, *ptrOr, blockStride, repeatStride, pregMask});
        Value newAddr = b.create<LLVM::PtrToIntOp>(loc, b.getI64Type(), call.getResult(0));
        mapUpdateNextPtrI64(storeOp.getNextPtr(), newAddr, valueMap);
        return success();
    }

    Value stride = valueMap.lookup(storeOp.getPostUpdateStride());
    if (!stride)
        stride = lookupOrCloneScalarValue(b, storeOp.getPostUpdateStride(), valueMap);
    if (!stride)
        return storeOp.emitError("failed to materialize post_update_stride in vector helper"), failure();
    FailureOr<Value> newPtrOr = emitVstsx1Post(b, loc, source, *ptrOr, stride, pregMask, sourceTy.getElementType());
    if (failed(newPtrOr))
        return storeOp.emitError("unsupported element type for tla.store_update: ") << sourceTy.getElementType(),
               failure();
    mapUpdateNextPtrI64(storeOp.getNextPtr(), b.create<LLVM::PtrToIntOp>(loc, b.getI64Type(), *newPtrOr), valueMap);
    return success();
}

static LogicalResult lowerStoreUnalignUpdate(
    ::tla::StoreUnalignUpdateOp op, OpBuilder& b, DenseMap<Value, Value>& valueMap)
{
    Location loc = op.getLoc();
    Value src = valueMap.lookup(op.getSource());
    if (!src)
        return failure();
    // AscendC: HIVM $eleSize = postUpdateStride×sizeof(T); next_ptr = HW newPtr.
    Value postUpdateStride = valueMap.lookup(op.getPostUpdateStride());
    if (!postUpdateStride)
        postUpdateStride = lookupOrCloneScalarValue(b, op.getPostUpdateStride(), valueMap);
    if (!postUpdateStride)
        return op.emitError("failed to materialize post_update_stride"), failure();
    auto ptrOr = resolveEffectiveUbPtr(op.getDest(), b, loc, valueMap);
    if (failed(ptrOr))
        return op.emitError("store_unalign_update requires a live UB address"), failure();
    auto srcTy = dyn_cast<VectorType>(src.getType());
    if (!srcTy)
        return failure();
    Type elemType = srcTy.getElementType();
    Value countBytes = strideElemsToByteOffI32(b, loc, postUpdateStride, elemType);
    // AscendC ABAB store: empty UnalignRegForStore + vstus.post + vstas (no vldas).
    Value localUreg = emitEmptyStoreUreg(b, loc);
    FailureOr<std::pair<Value, Value>> emitted = emitVstusPost(b, loc, src, *ptrOr, localUreg, countBytes, elemType);
    if (failed(emitted))
        return op.emitError("unsupported element type for tla.store_unalign_update"), failure();
    // Flush ureg at HW ptr after post-update (AscendC StoreUnAlignPost(..., 0)).
    if (failed(emitVstas(b, loc, emitted->first, emitted->second)))
        return op.emitError("failed to emit vstas for unalign store UPDATE"), failure();
    mapUpdateNextPtrI64(op.getNextPtr(), b.create<LLVM::PtrToIntOp>(loc, b.getI64Type(), emitted->second), valueMap);
    return success();
}

/// scf.for: rebuild the loop, including loop-carried iter_args, and lower its
/// body. Init args and the scf.yield operands may be register, index, or
/// scalar SSA values threaded through the helper.
static LogicalResult lowerScfFor(
    scf::ForOp forOp, OpBuilder& b, ModuleOp module, DenseMap<Value, Value>& valueMap,
    AutoFullPredicateCache& fullMasks)
{
    Location loc = forOp.getLoc();
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
    SmallVector<bool> wasPtr(regionIterArgs.size(), false);
    SmallVector<Value> mappedInits;
    for (auto [idx, init] : llvm::enumerate(forOp.getInitArgs())) {
        Value mapped = valueMap.lookup(init);
        if (!mapped)
            mapped = lookupOrCloneScalarValue(b, init, valueMap);
        if (!mapped)
            return forOp.emitError("failed to map scf.for init_arg #") << idx << " (type " << init.getType() << ")",
                   failure();
        wasIndex[idx] = isa<IndexType>(init.getType());
        wasPtr[idx] = isa<::tla::PtrType>(init.getType());
        if (wasIndex[idx] && isa<IndexType>(mapped.getType()))
            mapped = b.create<arith::IndexCastOp>(loc, i64Ty, mapped);
        // !tla.ptr carriers: helper carries i64. UPDATE lowering maps next_ptr
        // to i64; IntToPtr inits may still be !llvm.ptr — normalize so
        // init/yield types match. If valueMap still holds abstract !tla.ptr
        // (e.g. TensorDesc base not yet rewritten), rematerialize instead of
        // hard-failing the common path.
        if (wasPtr[idx]) {
            if (isa<::tla::PtrType>(mapped.getType())) {
                mapped = materializeTlaPtrCarrierAsI64(b, loc, init, valueMap);
                if (!mapped)
                    return forOp.emitError("scf.for !tla.ptr init_arg #")
                               << idx << " could not materialize to i64 (!llvm.ptr / inttoptr addr)",
                           failure();
            } else if (isa<LLVM::LLVMPointerType>(mapped.getType())) {
                mapped = b.create<LLVM::PtrToIntOp>(loc, i64Ty, mapped);
            } else if (!mapped.getType().isInteger(64)) {
                return forOp.emitError("scf.for !tla.ptr init_arg #") << idx << " must lower to i64 or !llvm.ptr",
                       failure();
            }
            valueMap[init] = mapped;
        }
        mappedInits.push_back(mapped);
    }

    LogicalResult bodyStatus = success();
    auto newFor = b.create<scf::ForOp>(
        loc, lb, ub, step, mappedInits, [&](OpBuilder& nb, Location nloc, Value iv, ValueRange iterArgs) {
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
                Value mapped = nestedMap.lookup(v);
                if (!mapped)
                    mapped = lookupOrCloneScalarValue(nb, v, nestedMap);
                if (!mapped) {
                    bodyStatus = failure();
                    break;
                }
                if (wasIndex[i] && isa<IndexType>(mapped.getType()))
                    mapped = nb.create<arith::IndexCastOp>(nloc, i64Ty, mapped);
                if (wasPtr[i]) {
                    if (isa<::tla::PtrType>(mapped.getType())) {
                        mapped = materializeTlaPtrCarrierAsI64(nb, nloc, v, nestedMap);
                        if (!mapped) {
                            bodyStatus = failure();
                            break;
                        }
                    } else if (isa<LLVM::LLVMPointerType>(mapped.getType())) {
                        mapped = nb.create<LLVM::PtrToIntOp>(nloc, i64Ty, mapped);
                    } else if (!mapped.getType().isInteger(64)) {
                        bodyStatus = failure();
                        break;
                    }
                    nestedMap[v] = mapped;
                }
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

/// scf.if: rebuild result-bearing conditionals after converting register
/// carrier types to physical builtin vectors. Each branch is lowered with an
/// independent value map, and its old scf.yield operands feed the new op.
static LogicalResult lowerScfIf(
    scf::IfOp ifOp, OpBuilder& b, ModuleOp module, DenseMap<Value, Value>& valueMap, AutoFullPredicateCache& fullMasks)
{
    Location loc = ifOp.getLoc();
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

static Value vfPrintCount(OpBuilder& b, Location loc, Value count, int64_t capacity)
{
    Value value = ::tla::castValueToI64(b, loc, count);
    Value zero = b.create<arith::ConstantIntOp>(loc, 0, 64);
    Value limit = b.create<arith::ConstantIntOp>(loc, capacity, 64);
    value = b.create<arith::MaxSIOp>(loc, value, zero);
    value = b.create<arith::MinSIOp>(loc, value, limit);
    return b.create<arith::TruncIOp>(loc, b.getI32Type(), value);
}

static LogicalResult emitVfDumpCall(
    OpBuilder& b, Location loc, ModuleOp module, StringRef kind, Type elementType, ValueRange operands)
{
    std::string diagnostic;
    auto suffix = getPrintTensorHelperSuffix(elementType, diagnostic);
    if (failed(suffix))
        return emitError(loc, diagnostic);
    std::string name = ("tla_vf_print_" + kind + "_" + *suffix).str();
    auto callee = module.lookupSymbol<func::FuncOp>(name);
    if (!callee) {
        OpBuilder mb(module.getBodyRegion());
        SmallVector<Type> types;
        for (Value operand : operands)
            types.push_back(operand.getType());
        callee = mb.create<func::FuncOp>(loc, name, b.getFunctionType(types, {}));
        annotateAivLibraryCall(callee);
        callee->setAttr("hacc.always_inline", b.getUnitAttr());
    }
    b.create<func::CallOp>(loc, callee, operands);
    return success();
}

static LogicalResult lowerVfRegisterPrint(
    ::tla::PrintVectorOp op, OpBuilder& b, ModuleOp module, DenseMap<Value, Value>& valueMap)
{
    Value value = valueMap.lookup(op.getValue());
    Value length = lookupOrCloneScalarValue(b, op.getLength(), valueMap);
    if (!value || !length)
        return op.emitError("cannot capture VF register print operands");
    auto type = cast<::tla::VectorSSAType>(op.getValue().getType());
    Value count = vfPrintCount(b, op.getLoc(), length, 2048 / type.getElementType().getIntOrFloatBitWidth());
    auto id = op->getAttrOfType<IntegerAttr>("tla.vf_print_id");
    if (!id)
        return op.emitError("VF register print call ID was not assigned");
    Value callId = b.create<arith::ConstantIntOp>(op.getLoc(), id.getInt(), 32);
    return emitVfDumpCall(b, op.getLoc(), module, "reg", type.getElementType(), {value, count, callId});
}

static LogicalResult lowerVfTensorPrint(
    ::tla::PrintTensorOp op, OpBuilder& b, ModuleOp module, DenseMap<Value, Value>& valueMap)
{
    auto desc = op.getValue().getDefiningOp<::tla::TensorDescOp>();
    auto type = cast<::tla::TlaTensorType>(op.getValue().getType());
    if (!desc || type.getPtr().getAddrspace() != AddressSpace::ub)
        return op.emitError("tensor printing inside a SIMD vec.func requires UB storage");
    Value value = valueMap.lookup(op.getValue());
    Value length = lookupOrCloneScalarValue(b, op.getLength(), valueMap);
    if (!value || !length)
        return op.emitError("cannot capture VF tensor print operands");
    Location loc = op.getLoc();
    int64_t rank = op.getShape().size();
    SmallVector<Value> shape;
    auto layout = type.getLayout().getLayoutTag();
    bool packed = layout != LayoutTag::RowMajor && layout != LayoutTag::ColumnMajor;
    for (unsigned i = 0; i < rank; ++i) {
        Value extent;
        if (op.getShape()[i] >= 0)
            extent = b.create<arith::ConstantIntOp>(loc, op.getShape()[i], 64);
        else {
            Value original = rank == 1 ? desc.getShape1() :
                             packed    ? (i == 0 ? desc.getOriginShape0() : desc.getOriginShape1()) :
                                         (i == 0 ? desc.getShape0() : desc.getShape1());
            extent = lookupOrCloneScalarValue(b, original, valueMap);
            if (!extent)
                return op.emitError("cannot capture VF tensor print shape");
            extent = ::tla::castValueToI64(b, loc, extent);
        }
        shape.push_back(extent);
    }
    Value elements = shape.front();
    if (rank == 2)
        elements = b.create<arith::MulIOp>(loc, elements, shape.back());
    length = b.create<arith::MinSIOp>(loc, length, elements);
    Type elementType = type.getPtr().getPointee();
    Value count = vfPrintCount(b, loc, length, 1952 * 8 / elementType.getIntOrFloatBitWidth());
    Value shape0 = b.create<arith::TruncIOp>(loc, b.getI32Type(), shape.front());
    Value shape1 = rank == 2 ? Value(b.create<arith::TruncIOp>(loc, b.getI32Type(), shape.back())) :
                               Value(b.create<arith::ConstantIntOp>(loc, 0, 32));
    Value rankValue = b.create<arith::ConstantIntOp>(loc, rank, 32);
    auto id = op->getAttrOfType<IntegerAttr>("tla.vf_print_id");
    if (!id)
        return op.emitError("VF tensor print call ID was not assigned");
    Value callId = b.create<arith::ConstantIntOp>(loc, id.getInt(), 32);
    return emitVfDumpCall(b, loc, module, "ub", elementType, {value, count, shape0, shape1, rankValue, callId});
}

static LogicalResult lowerNestedVectorOp(
    Operation& op, OpBuilder& b, ModuleOp module, DenseMap<Value, Value>& valueMap, AutoFullPredicateCache& fullMasks)
{
    Location loc = op.getLoc();

    // make_shape / make_coord are dead after tla-lower-tensor-desc (their leaves
    // were folded into tensor_desc operands); skip them. (tla-finalize-memref
    // erases them.)
    if (isa<::tla::MakeShapeOp, ::tla::MakeCoordOp, ::tla::MakeStrideOp, ::tla::MakeLayoutOp>(op))
        return success();

    if (auto constant = dyn_cast<arith::ConstantOp>(op)) {
        valueMap[constant.getResult()] = b.clone(op)->getResult(0);
        return success();
    }

    if (auto call = dyn_cast<::tla::CallExternOp>(op)) {
        SmallVector<Value> args;
        for (Value operand : call.getOperands()) {
            Value mapped = lookupOrCloneScalarValue(b, operand, valueMap);
            if (!mapped)
                return call.emitOpError("failed to map extern argument into the VF");
            args.push_back(mapped);
        }
        auto callee = module.lookupSymbol<func::FuncOp>(call.getCallee());
        if (!callee)
            return call.emitOpError("failed to find extern callee in module: ") << call.getCallee();
        auto lowered = b.create<func::CallOp>(loc, callee, args);
        for (auto [source, result] : llvm::zip(call.getResults(), lowered.getResults()))
            valueMap[source] = result;
        return success();
    }

    if (auto ptrToInt = dyn_cast<::tla::PtrToIntOp>(op)) {
        Value mapped = lookupOrCloneScalarValue(b, ptrToInt.getPtr(), valueMap);
        if (!mapped)
            return ptrToInt.emitError("failed to map tla.ptrtoint operand in vector helper"), failure();
        if (mapped.getType().isInteger(64))
            valueMap[ptrToInt.getResult()] = mapped;
        else if (isa<LLVM::LLVMPointerType>(mapped.getType()))
            valueMap[ptrToInt.getResult()] = b.create<LLVM::PtrToIntOp>(loc, b.getI64Type(), mapped);
        else
            return ptrToInt.emitError("expected i64 or !llvm.ptr for tla.ptrtoint"), failure();
        return success();
    }

    // Ptr-backed tiles: storage-identity outlines data memrefs onto the seed
    // UB helper arg for ABI dedup; the inttoptr still materializes the live
    // ``!llvm.ptr<6>`` so effective-address materialization / UPDATE `.post`
    // / SCF yield share end-to-end SSA.
    if (auto intToPtr = dyn_cast<::tla::IntToPtrOp>(op)) {
        Value addr = lookupOrCloneScalarValue(b, intToPtr.getAddr(), valueMap);
        if (!addr)
            return intToPtr.emitError("failed to map tla.inttoptr address in vector helper"), failure();
        valueMap[intToPtr.getResult()] = intToUbPtr(b, loc, addr);
        return success();
    }

    // tla.tensor_desc: materialize it as a subview of the helper's base memref
    // argument (see materializeTensorDescSubview).
    if (auto descOp = dyn_cast<::tla::TensorDescOp>(op))
        return materializeTensorDescSubview(descOp, b, valueMap);

    if (auto loadOp = dyn_cast<::tla::LoadOp>(op))
        return lowerLoad(loadOp, b, module, valueMap, fullMasks);

    if (auto loadUpdateOp = dyn_cast<::tla::LoadUpdateOp>(op))
        return lowerLoadUpdate(loadUpdateOp, b, module, valueMap, fullMasks);
    if (auto printOp = dyn_cast<::tla::DebugPrintOp>(op)) {
        SmallVector<Value> values;
        for (Value value : printOp.getValues()) {
            Value mapped = lookupOrCloneScalarValue(b, value, valueMap);
            if (!mapped)
                return printOp.emitError("cannot capture VF print operand");
            values.push_back(mapped);
        }
        return lowerVfScalarPrint(b, printOp, values);
    }
    if (auto printOp = dyn_cast<::tla::PrintVectorOp>(op))
        return lowerVfRegisterPrint(printOp, b, module, valueMap);
    if (auto printOp = dyn_cast<::tla::PrintTensorOp>(op))
        return lowerVfTensorPrint(printOp, b, module, valueMap);

    if (auto loadUnalignUpdateOp = dyn_cast<::tla::LoadUnalignUpdateOp>(op))
        return lowerLoadUnalignUpdate(loadUnalignUpdateOp, b, valueMap);

    if (auto storeUnalignUpdateOp = dyn_cast<::tla::StoreUnalignUpdateOp>(op))
        return lowerStoreUnalignUpdate(storeUnalignUpdateOp, b, valueMap);

    // Preserve local-memory ordering while outlining a vec.func into its AVE
    // helper. Every operation in the source region must be recreated explicitly;
    // otherwise helper construction fails and leaves the whole vec.func behind.
    if (auto localMemBarOp = dyn_cast<::tla::LocalMemBarOp>(op)) {
        return lowerLocalMemBar(b, localMemBarOp);
    }

    if (auto bitCastOp = dyn_cast<::tla::BitCastOp>(op))
        return lowerBitCast(bitCastOp, b, valueMap);

    if (auto castOp = dyn_cast<::tla::CastOp>(op))
        return lowerCast(castOp, b, module, valueMap, fullMasks);

    if (auto fullOp = dyn_cast<::tla::FullOp>(op))
        return lowerFull(fullOp, b, module, valueMap, fullMasks);

    if (auto arangeOp = dyn_cast<::tla::ArangeOp>(op))
        return lowerArange(arangeOp, b, module, valueMap, fullMasks);

    if (auto info = getVectorBinaryInfo(&op))
        return lowerVectorBinary(op, &*info, b, module, valueMap, fullMasks);

    if (auto info = getVectorScalarBinaryInfo(&op))
        return lowerVectorScalarBinary(op, &*info, b, module, valueMap, fullMasks);

    // tla.where: per-lane select. The mask controls which lanes take `x`; the
    // remaining lanes take `y`. VectorSSA lowers to ave.hir.vsel(mask, x, y),
    // while MaskSSA calls the linked psel bitcode helper.
    if (auto whereOp = dyn_cast<::tla::WhereOp>(op)) {
        Value mask = valueMap.lookup(whereOp.getMask());
        Value x = valueMap.lookup(whereOp.getX());
        Value y = valueMap.lookup(whereOp.getY());
        if (!mask || !x || !y)
            return failure();

        // After type conversion all MaskSSA values use vector<256xi1>, so the
        // original TLA operand type determines whether this is predicate select.
        if (isa<::tla::MaskSSAType>(whereOp.getX().getType())) {
            VectorType pregType = fullPregVecType(b.getContext());
            Value pregMask = castMaskToPregType(b, loc, mask, pregType);
            Value pregX = castMaskToPregType(b, loc, x, pregType);
            Value pregY = castMaskToPregType(b, loc, y, pregType);
            auto callee = getOrCreateMaskSelectLibraryCall(module, loc, pregType);
            valueMap[whereOp.getResult()] =
                b.create<func::CallOp>(loc, callee, ValueRange{pregX, pregY, pregMask}).getResult(0);
            return success();
        }

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

    if (auto squeezeOp = dyn_cast<::tla::SqueezeOp>(op))
        return lowerSqueeze(squeezeOp, b, module, valueMap, fullMasks);

    // tla.histograms: Reg FREQUENCY/ACCUMULATE histogram via linked bitcode
    // (dhistv2 / chistv2). Not lowered through HIVMAVE.
    if (auto histOp = dyn_cast<::tla::HistogramsOp>(op)) {
        Value src = valueMap.lookup(histOp.getSrc());
        if (!src)
            return failure();
        auto srcTy = dyn_cast<VectorType>(src.getType());
        if (!srcTy)
            return failure();
        auto srcElem = dyn_cast<IntegerType>(srcTy.getElementType());
        if (!srcElem || srcElem.getWidth() != 8)
            return histOp.emitError("tla.histograms src must be i8/ui8"), failure();

        auto histResultSSA = dyn_cast<VectorSSAType>(histOp.getResult().getType());
        if (!histResultSSA)
            return failure();
        Type resultElemType = histResultSSA.getElementType();
        auto resultElemInt = dyn_cast<IntegerType>(resultElemType);
        if (!resultElemInt || resultElemInt.getWidth() != 16)
            return histOp.emitError("tla.histograms result must be i16/ui16"), failure();

        auto srcCtx = deriveVecCtxForElement(srcElem);
        auto dstCtx = deriveVecCtxForElement(resultElemType);
        if (failed(srcCtx) || failed(dstCtx))
            return failure();

        std::string calleeName =
            getHistogramsLibraryCallName(histOp.getMode().getHistogramsBin(), histOp.getHistType().getHistogramsType());
        VectorType pregVecType = fullPregVecType(b.getContext());
        Value mask;
        if (histOp.getMask()) {
            mask = valueMap.lookup(histOp.getMask());
            if (!mask)
                return failure();
        } else {
            mask = fullMasks.get(loc, srcCtx->lanes);
        }
        Value preg = castMaskToPregType(b, loc, mask, pregVecType);

        // dst is required (in-place accumulator). Callers zero with
        // tla.full(0, ...) when starting a fresh histogram; no Pass-side zeroing.
        Value dst = valueMap.lookup(histOp.getDst());
        if (!dst)
            return failure();

        VectorType srcU8Type =
            VectorType::get(srcCtx->vecType.getShape(), IntegerType::get(b.getContext(), 8, IntegerType::Unsigned));
        VectorType dstU16Type =
            VectorType::get(dstCtx->vecType.getShape(), IntegerType::get(b.getContext(), 16, IntegerType::Unsigned));
        if (src.getType() != srcU8Type)
            src = b.create<UnrealizedConversionCastOp>(loc, srcU8Type, src).getResult(0);
        if (dst.getType() != dstU16Type)
            dst = b.create<UnrealizedConversionCastOp>(loc, dstU16Type, dst).getResult(0);

        auto callee = getOrCreateHistogramsLibraryCall(module, loc, srcU8Type, dstU16Type, pregVecType, calleeName);
        Value result = b.create<func::CallOp>(loc, callee, ValueRange{src, dst, preg}).getResult(0);
        if (result.getType() != dstCtx->vecType)
            result = b.create<UnrealizedConversionCastOp>(loc, dstCtx->vecType, result).getResult(0);
        valueMap[histOp.getResult()] = result;
        return success();
    }

    if (auto reduceOp = dyn_cast<::tla::ReduceOp>(op))
        return lowerReduce(reduceOp, b, module, valueMap, fullMasks);

    if (auto rdbOp = dyn_cast<::tla::ReduceIntraBlockOp>(op))
        return lowerReduceIntraBlock(rdbOp, b, module, valueMap, fullMasks);

    if (auto packOp = dyn_cast<::tla::PackOp>(op))
        return lowerPack(packOp, b, module, valueMap);
    if (auto unpackOp = dyn_cast<::tla::UnpackOp>(op))
        return lowerUnpack(unpackOp, b, module, valueMap);

    if (auto interleaveOp = dyn_cast<::tla::InterleaveOp>(op))
        return lowerInterleave(interleaveOp, b, module, valueMap, fullMasks);

    if (auto deInterleaveOp = dyn_cast<::tla::DeInterleaveOp>(op))
        return lowerDeInterleave(deInterleaveOp, b, module, valueMap, fullMasks);

    if (auto gatherOp = dyn_cast<::tla::GatherOp>(op))
        return lowerGather(gatherOp, b, module, valueMap, fullMasks);

    if (auto info = getVectorUnaryInfo(&op))
        return lowerVectorUnary(op, &*info, b, module, valueMap, fullMasks);

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

    if (auto updateMaskOp = dyn_cast<::tla::UpdateMaskOp>(op))
        return lowerUpdateMask(updateMaskOp, b, module, valueMap, fullMasks);

    if (auto cmpOp = dyn_cast<::tla::CmpOp>(op))
        return lowerCmp(cmpOp, b, module, valueMap, fullMasks);

    if (auto storeOp = dyn_cast<::tla::StoreOp>(op))
        return lowerStore(storeOp, b, module, valueMap, fullMasks);

    if (auto storeUpdateOp = dyn_cast<::tla::StoreUpdateOp>(op))
        return lowerStoreUpdate(storeUpdateOp, b, module, valueMap, fullMasks);

    if (auto forOp = dyn_cast<scf::ForOp>(op))
        return lowerScfFor(forOp, b, module, valueMap, fullMasks);

    if (auto ifOp = dyn_cast<scf::IfOp>(op))
        return lowerScfIf(ifOp, b, module, valueMap, fullMasks);

    // Index/scalar arithmetic (arith.*): clone with mapped operands. This covers
    // both the arithmetic feeding offsets and conditions, and the coord/origin
    // arithmetic (addi/subi/minsi) that tla-lower-tensor-desc emits for
    // dynamic-coord tile_views. Static coords fold to constants (createOrFold)
    // and are handled by the arith::ConstantOp arm above.
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
    if (isa<mlir::memref::ExtractStridedMetadataOp, mlir::memref::ReinterpretCastOp,
            mlir::memref::ExtractAlignedPointerAsIndexOp>(op)) {
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

    // Keep profiling boundaries at their original position inside the helper.
    // The late profiling pass, not outlining, materializes hardware controls.
    if (isa<::tla::ProfStartOp, ::tla::ProfEndOp>(op)) {
        b.clone(op);
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
        if (failed(lowerNestedVectorOp(op, b, module, valueMap, fullMasks))) {
            // Avoid emitOpError: printing a half-rewritten op can assert in AsmPrinter.
            return op.emitError("unsupported op while outlining tla.vec.func into vector helper: ") << op.getName(),
                   failure();
        }
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

/// Chase ``scf.for`` / ``scf.while`` iter_args *and* results back to their init
/// so ``tensor_desc(inttoptr(scf_i64))`` shares one helper memref *identity*
/// with the seed tile (parent UB). This is allocation dedup / call-site ABI
/// only — not the effective address used by ordinary load/store or UPDATE.
/// Post-loop epilogue tiles carry for/while *results*; without chasing those,
/// helper-call materialization would take an address defined inside
/// ``tla.vec.func`` and leave an external use that blocks eraseOp.
static Value chaseScfCarrierInit(Value address)
{
    address = stripHelperAddressCasts(address);
    // No fixed step budget: stop at a fixed point or SSA cycle (visited).
    DenseSet<Value> visited;
    while (visited.insert(address).second) {
        if (auto ba = dyn_cast<BlockArgument>(address)) {
            Operation* parent = ba.getOwner()->getParentOp();
            if (auto forOp = dyn_cast<scf::ForOp>(parent)) {
                // Region args: induction var then iter_args.
                unsigned argNum = ba.getArgNumber();
                if (argNum == 0)
                    break;
                unsigned iterIdx = argNum - 1;
                if (iterIdx >= forOp.getNumRegionIterArgs())
                    break;
                address = stripHelperAddressCasts(forOp.getInitArgs()[iterIdx]);
                continue;
            }
            if (auto whileOp = dyn_cast<scf::WhileOp>(parent)) {
                unsigned argNum = ba.getArgNumber();
                if (argNum >= whileOp.getInits().size())
                    break;
                address = stripHelperAddressCasts(whileOp.getInits()[argNum]);
                continue;
            }
            break;
        }
        // Epilogue after the loop: carrier is an scf.for / scf.while result.
        if (auto forOp = address.getDefiningOp<scf::ForOp>()) {
            unsigned idx = cast<OpResult>(address).getResultNumber();
            if (idx >= forOp.getNumRegionIterArgs())
                break;
            address = stripHelperAddressCasts(forOp.getInitArgs()[idx]);
            continue;
        }
        if (auto whileOp = address.getDefiningOp<scf::WhileOp>()) {
            unsigned idx = cast<OpResult>(address).getResultNumber();
            if (idx >= whileOp.getInits().size())
                break;
            address = stripHelperAddressCasts(whileOp.getInits()[idx]);
            continue;
        }
        break;
    }
    return address;
}

// Allocation identity for helper-operand dedup. Must not be used as the
// effective load/store address (live inttoptr / descriptor coord own that).
// Chase UPDATE next_ptr producers, tile_view, and ptrtoint/inttoptr / SCF
// carrier wrappers back to the same seed so chained UPDATE windows share one
// helper memref. Do not chase ``tla.ptr_add`` or peel lowered i64 addi/muli.
static Value getVectorHelperStorageIdentity(Value source)
{
    // IR-wrapper chase only (not loop trip count). Unbounded except SSA cycles.
    DenseSet<Value> visited;
    while (visited.insert(source).second) {
        source = stripHelperAddressCasts(source);
        // Tile SSA carried across ``scf.for`` / ``scf.while`` (iter_arg or result)
        // must share the seed identity with the init window. Otherwise a
        // once→for→once UnAlign stream registers the post-loop tile as a new
        // helper operand whose ``inttoptr`` addr is an in-region ``ptrtoint`` of
        // the live next_ptr — dominate check rejects the outline (silent
        // legalize failure). Same chase as ``chaseScfCarrierInit`` but on the
        // tensor/ptr carrier itself, not only the i64 address.
        if (auto ba = dyn_cast<BlockArgument>(source)) {
            Operation* parent = ba.getOwner()->getParentOp();
            if (auto forOp = dyn_cast<scf::ForOp>(parent)) {
                unsigned argNum = ba.getArgNumber();
                if (argNum > 0 && argNum - 1 < forOp.getNumRegionIterArgs()) {
                    source = forOp.getInitArgs()[argNum - 1];
                    continue;
                }
            } else if (auto whileOp = dyn_cast<scf::WhileOp>(parent)) {
                unsigned argNum = ba.getArgNumber();
                if (argNum < whileOp.getInits().size()) {
                    source = whileOp.getInits()[argNum];
                    continue;
                }
            }
            break;
        }
        if (auto forOp = source.getDefiningOp<scf::ForOp>()) {
            unsigned idx = cast<OpResult>(source).getResultNumber();
            if (idx < forOp.getNumRegionIterArgs()) {
                source = forOp.getInitArgs()[idx];
                continue;
            }
            break;
        }
        if (auto whileOp = source.getDefiningOp<scf::WhileOp>()) {
            unsigned idx = cast<OpResult>(source).getResultNumber();
            if (idx < whileOp.getInits().size()) {
                source = whileOp.getInits()[idx];
                continue;
            }
            break;
        }
        if (auto descOp = source.getDefiningOp<::tla::TensorDescOp>()) {
            source = descOp.getBase();
            continue;
        }
        if (auto makeOp = source.getDefiningOp<::tla::MakeTensorOp>()) {
            source = makeOp.getPtr();
            continue;
        }
        if (auto makeLikeOp = source.getDefiningOp<::tla::MakeTensorLikeOp>()) {
            source = makeLikeOp.getPtr();
            continue;
        }
        if (auto tileOp = source.getDefiningOp<::tla::TileViewOp>()) {
            source = tileOp.getSource();
            continue;
        }
        if (auto tensorPtr = source.getDefiningOp<::tla::TensorPtrOp>()) {
            source = tensorPtr.getSrc();
            continue;
        }
        if (auto loadUpdate = source.getDefiningOp<::tla::LoadUpdateOp>()) {
            source = loadUpdate.getSource();
            continue;
        }
        if (auto loadUnalignUpdate = source.getDefiningOp<::tla::LoadUnalignUpdateOp>()) {
            source = loadUnalignUpdate.getSource();
            continue;
        }
        if (auto storeUpdate = source.getDefiningOp<::tla::StoreUpdateOp>()) {
            source = storeUpdate.getDest();
            continue;
        }
        if (auto storeUnalignUpdate = source.getDefiningOp<::tla::StoreUnalignUpdateOp>()) {
            source = storeUnalignUpdate.getDest();
            continue;
        }
        if (auto ptrToInt = source.getDefiningOp<::tla::PtrToIntOp>()) {
            source = ptrToInt.getPtr();
            continue;
        }
        if (auto llvmPtrToInt = source.getDefiningOp<LLVM::PtrToIntOp>()) {
            source = llvmPtrToInt.getArg();
            continue;
        }
        if (auto intToPtr = source.getDefiningOp<::tla::IntToPtrOp>()) {
            // Address SSA is i64; chase SCF carriers then unwrap ptrtoint seeds.
            Value addr = chaseScfCarrierInit(intToPtr.getAddr());
            if (auto fromPtr = addr.getDefiningOp<::tla::PtrToIntOp>()) {
                source = fromPtr.getPtr();
                continue;
            }
            if (auto fromPtr = addr.getDefiningOp<LLVM::PtrToIntOp>()) {
                source = fromPtr.getArg();
                continue;
            }
            return addr;
        }
        if (auto llvmIntToPtr = source.getDefiningOp<LLVM::IntToPtrOp>()) {
            Value addr = chaseScfCarrierInit(llvmIntToPtr.getArg());
            if (auto fromPtr = addr.getDefiningOp<::tla::PtrToIntOp>()) {
                source = fromPtr.getPtr();
                continue;
            }
            if (auto fromPtr = addr.getDefiningOp<LLVM::PtrToIntOp>()) {
                source = fromPtr.getArg();
                continue;
            }
            return addr;
        }
        if (auto castOp = source.getDefiningOp<mlir::memref::CastOp>()) {
            source = castOp.getSource();
            continue;
        }
        if (auto pointerCast = source.getDefiningOp<hivm::PointerCastOp>())
            return stripHelperAddressCasts(pointerCast.getSingleAddr());
        if (source.getDefiningOp<::tla::AllocPtrOp>())
            return source;
        break;
    }
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
        if (identity != getVectorHelperStorageIdentity(operand))
            continue;
        FailureOr<MemRefType> operandType = getVectorHelperOperandType(operand);
        // Same allocation seed: reuse the representative even when ``source`` is
        // an UPDATE next-window tile whose raw next_ptr cannot type a helper
        // memref by itself (typed via the seed inttoptr instead).
        if (succeeded(operandType) && (failed(sourceType) || *sourceType == *operandType)) {
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
        if (auto loadUpdateOp = dyn_cast<::tla::LoadUpdateOp>(op)) {
            addVectorHelperOperand(loadUpdateOp.getSource(), operands, aliases);
            continue;
        }
        if (auto loadUnalignUpdateOp = dyn_cast<::tla::LoadUnalignUpdateOp>(op)) {
            addVectorHelperOperand(loadUnalignUpdateOp.getSource(), operands, aliases);
            continue;
        }
        if (auto storeOp = dyn_cast<::tla::StoreOp>(op)) {
            addVectorHelperOperand(storeOp.getDest(), operands, aliases);
            continue;
        }
        if (auto storeUpdateOp = dyn_cast<::tla::StoreUpdateOp>(op)) {
            addVectorHelperOperand(storeUpdateOp.getDest(), operands, aliases);
            continue;
        }
        if (auto storeUnalignUpdateOp = dyn_cast<::tla::StoreUnalignUpdateOp>(op)) {
            addVectorHelperOperand(storeUnalignUpdateOp.getDest(), operands, aliases);
            continue;
        }
        if (auto printOp = dyn_cast<::tla::PrintTensorOp>(op)) {
            addVectorHelperOperand(printOp.getValue(), operands, aliases);
            continue;
        }
        if (auto gatherOp = dyn_cast<::tla::GatherOp>(op)) {
            // Only UB-tensor gather needs a helper memref argument.
            if (isa<::tla::TlaTensorType>(gatherOp.getX().getType()))
                addVectorHelperOperand(gatherOp.getX(), operands, aliases);
            continue;
        }
        // A memref-backed tensor.ptr in an extern argument reads the aligned address.
        if (auto addr = dyn_cast<mlir::memref::ExtractAlignedPointerAsIndexOp>(op)) {
            auto source = getScalarAccessHelperMemref(addr.getSource(), outlinedRegion, callSite, dominanceInfo);
            if (failed(source))
                return failure();
            addVectorHelperOperand(*source, operands, aliases);
            continue;
        }
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
    if (!body)
        return vecFuncOp.emitOpError("vector helper needs a body"), failure();

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
            return vecFuncOp.emitOpError("failed to type helper memref operand"), failure();
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
    // Do not stamp llvm.noalias on helper memref args: bump-plan disjointness
    // is a performance assumption without independent alias proof, and must
    // not be a correctness dependency. Same-base in-place load/store already
    // share one argument via helperOperandAliases.

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
    DenseMap<Value, Value> identityToArg;
    for (Value operand : helperOperands) {
        if (Value arg = valueMap.lookup(operand)) {
            identityToArg[getVectorHelperStorageIdentity(operand)] = arg;
            if (auto d = operand.getDefiningOp<::tla::TensorDescOp>())
                baseToArg[d.getBase()] = arg;
        }
    }
    vecFuncOp.walk([&](::tla::TensorDescOp desc) {
        if (valueMap.count(desc.getResult()))
            return;
        if (auto it = baseToArg.find(desc.getBase()); it != baseToArg.end()) {
            valueMap[desc.getResult()] = it->second;
            return;
        }
        // Distinct ``inttoptr(scf_i64)`` ops share chased seed identity.
        if (auto it = identityToArg.find(getVectorHelperStorageIdentity(desc.getResult())); it != identityToArg.end())
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
            vecFuncOp.emitOpError("failed to materialize external tensor_desc into vector helper");
            helper.erase();
            return failure();
        }
    }

    AutoFullPredicateCache fullMasks(&helper.getBody().front());
    if (failed(lowerNestedVectorBlock(body, b, module, valueMap, fullMasks))) {
        // Discard the partially-built helper so an unsupported construct fails
        // cleanly (the vec.func is left intact) instead of leaking malformed IR.
        vecFuncOp.emitOpError("failed to lower nested vector body into helper");
        helper.erase();
        return failure();
    }
    b.create<func::ReturnOp>(vectorOp->getLoc());
    return helper;
}

// tla.vec.func carries its execution model in a "mode" string attribute
// ("simd" / "simt", either case). SIMD is the historical default and is assumed
// when the attribute is absent.

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
        SmallVector<::tla::LoadUpdateOp, 4> loadUpdates;
        SmallVector<::tla::FullOp, 4> fulls;
        SmallVector<::tla::CreateMaskOp, 4> createMasks;
        SmallVector<::tla::UpdateMaskOp, 4> updateMasks;
        SmallVector<::tla::ArangeOp, 4> aranges;
        SmallVector<Operation*, 4> computeOps;
        SmallVector<::tla::StoreOp, 2> stores;
        SmallVector<::tla::StoreUpdateOp, 2> storeUpdates;
        SmallVector<Value, 4> unalignLoads;
        int unalignStores = 0;
        bool hasDebugPrint = false;
        bool hasExtern = false;
        vecFuncOp->walk([&](Operation* op) {
            hasDebugPrint |= isa<::tla::DebugPrintOp, ::tla::PrintVectorOp, ::tla::PrintTensorOp>(op);
            if (isa<::tla::CallExternOp>(op)) {
                hasExtern = true;
            } else if (auto load = dyn_cast<::tla::LoadOp>(op)) {
                loads.push_back(load);
            } else if (auto loadUpdate = dyn_cast<::tla::LoadUpdateOp>(op)) {
                loadUpdates.push_back(loadUpdate);
            } else if (auto loadUnalignUpdate = dyn_cast<::tla::LoadUnalignUpdateOp>(op)) {
                unalignLoads.push_back(loadUnalignUpdate.getResult());
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
            } else if (auto storeUpdate = dyn_cast<::tla::StoreUpdateOp>(op)) {
                storeUpdates.push_back(storeUpdate);
            } else if (isa<::tla::StoreUnalignUpdateOp>(op)) {
                ++unalignStores;
            } else if (isVectorComputeOp(op)) {
                computeOps.push_back(op);
            }
            return WalkResult::advance();
        });
        if (hasDebugPrint && module->hasAttr("tla.ub_reserve_released"))
            return vecFuncOp.emitError("VF printing requires the default reserved UB");
        if (!hasExtern && stores.empty() && storeUpdates.empty() && unalignStores == 0 && !hasDebugPrint) {
            // Scalar-only (or empty) VF cannot be outlined as a BiSheng helper — that
            // path requires tla.store or tla.store_update. If there is also no tile
            // load/compute, inline the body into the parent (same as tla.vector
            // flattening) so scalar_load/store + scf stay legal for later
            // convert-scf-to-cf.
            // tla.call_extern is opaque, it may have side effects, so if
            // there is no opaque extern or vector stores, flatten scalar-only
            // regions.
            if (!loads.empty() || !loadUpdates.empty() || !unalignLoads.empty() || !fulls.empty() ||
                !createMasks.empty() || !updateMasks.empty() || !aranges.empty() || !computeOps.empty())
                return rewriter.notifyMatchFailure(
                    vecFuncOp, "expected tla.vec.func body with a tla.store or tla.store_update");
            rewriter.inlineBlockBefore(body, vecFuncOp->getBlock(), vecFuncOp->getIterator());
            rewriter.eraseOp(vecFuncOp);
            return success();
        }

        // Validate the graph: every compute operand and store source must come from
        // a tla.load result or a prior compute result inside this region.
        DenseSet<Value> producedValues;
        vecFuncOp.walk([&](::tla::CallExternOp call) {
            producedValues.insert(call.getResults().begin(), call.getResults().end());
        });
        for (::tla::LoadOp load : loads) {
            producedValues.insert(load.getResult());
            if (Value result2 = load.getResult2())
                producedValues.insert(result2);
        }
        for (::tla::LoadUpdateOp loadUpdate : loadUpdates)
            producedValues.insert(loadUpdate.getResult());
        for (Value unalignLoad : unalignLoads)
            producedValues.insert(unalignLoad);
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
                        vecFuncOp, "expected tla.where operands from legal vector or mask producers");
            } else if (auto squeezeOp = dyn_cast<::tla::SqueezeOp>(computeOp)) {
                if (!producedValues.contains(squeezeOp.getSrc()))
                    return rewriter.notifyMatchFailure(
                        vecFuncOp, "expected tla.squeeze src from tla.load or prior compute op");
                if (!producedValues.contains(squeezeOp.getMask()))
                    return rewriter.notifyMatchFailure(
                        vecFuncOp,
                        "expected tla.squeeze mask from create/update mask or "
                        "prior mask compute op");
            } else if (auto histOp = dyn_cast<::tla::HistogramsOp>(computeOp)) {
                if (!producedValues.contains(histOp.getSrc()))
                    return rewriter.notifyMatchFailure(
                        vecFuncOp, "expected tla.histograms src from tla.load or prior compute op");
                if (!producedValues.contains(histOp.getDst()))
                    return rewriter.notifyMatchFailure(
                        vecFuncOp, "expected tla.histograms dst from tla.full, tla.load, or prior compute op");
                if (histOp.getMask() && !producedValues.contains(histOp.getMask()))
                    return rewriter.notifyMatchFailure(
                        vecFuncOp,
                        "expected tla.histograms mask from create/update mask or "
                        "prior mask compute op");
            } else if (auto reduceOp = dyn_cast<::tla::ReduceOp>(computeOp)) {
                Value operand = reduceOp.getOperand();
                if (reduceOp.getMask() && !producedValues.contains(reduceOp.getMask()))
                    return rewriter.notifyMatchFailure(
                        vecFuncOp, "expected tla.reduce mask from a legal mask producer");
                if (!producedValues.contains(operand))
                    return rewriter.notifyMatchFailure(
                        vecFuncOp, "expected tla.reduce operand from tla.load or prior compute op");
            } else if (auto reduceDataBlockOp = dyn_cast<::tla::ReduceIntraBlockOp>(computeOp)) {
                if (!producedValues.contains(reduceDataBlockOp.getMask()))
                    return rewriter.notifyMatchFailure(
                        vecFuncOp, "expected tla.reduce_intra_block mask from a legal mask producer");
                if (!producedValues.contains(reduceDataBlockOp.getOperand()))
                    return rewriter.notifyMatchFailure(
                        vecFuncOp, "expected tla.reduce_intra_block operand from tla.load or prior compute op");
            } else if (auto packOp = dyn_cast<::tla::PackOp>(computeOp)) {
                if (!producedValues.contains(packOp.getSrc()))
                    return rewriter.notifyMatchFailure(
                        vecFuncOp, "expected tla.pack src from a legal register producer");
            } else if (auto unpackOp = dyn_cast<::tla::UnpackOp>(computeOp)) {
                if (!producedValues.contains(unpackOp.getSrc()))
                    return rewriter.notifyMatchFailure(
                        vecFuncOp, "expected tla.unpack src from a legal register producer");
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
                if (isa<::tla::VectorSSAType>(gatherOp.getX().getType())) {
                    if (!producedValues.contains(gatherOp.getX()))
                        return rewriter.notifyMatchFailure(
                            vecFuncOp, "expected tla.gather src from tla.load or prior compute op");
                }
                if (gatherOp.getMask() && !producedValues.contains(gatherOp.getMask()))
                    return rewriter.notifyMatchFailure(
                        vecFuncOp, "expected tla.gather mask from a legal mask producer");
                if (!producedValues.contains(gatherOp.getY()))
                    return rewriter.notifyMatchFailure(
                        vecFuncOp, "expected tla.gather y operand from tla.load or prior compute op");
            } else if (auto bitCastOp = dyn_cast<::tla::BitCastOp>(computeOp)) {
                if (!producedValues.contains(bitCastOp.getSource()))
                    return rewriter.notifyMatchFailure(
                        vecFuncOp, "expected tla.bitcast source from tla.load or prior compute op");
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
        for (::tla::StoreUpdateOp storeUpdate : storeUpdates) {
            if (storeUpdate.getMask() && !producedValues.contains(storeUpdate.getMask()))
                return rewriter.notifyMatchFailure(
                    vecFuncOp, "expected tla.store_update mask from a legal mask producer");
            if (!producedValues.contains(storeUpdate.getSource()))
                return rewriter.notifyMatchFailure(
                    vecFuncOp, "expected tla.store_update source from tla.load_update or compute op");
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
        if (helperOperands.empty() && !hasDebugPrint && !hasExtern)
            return rewriter.notifyMatchFailure(vecFuncOp, "expected vector region tensor operands");
        // Scalars captured from outside the region (e.g. a sub_block_idx computed at
        // the top of the kernel) are passed as trailing scalar arguments.
        SmallVector<Value> scalarOperands;
        collectVectorHelperScalarOperands(vecFuncOp, scalarOperands);

        // Call-site ``hivm.pointer_cast`` consumes the inttoptr *address* SSA. If
        // that address is defined inside ``tla.vec.func`` (e.g. lowered
        // ``tla.ptr_add`` as ``addi``), materializing the helper operand would
        // create an external use and ``eraseOp(vecFuncOp)`` would assert. Reject
        // the outline with a match failure so later legalization reports a
        // normal diagnostic (same shape as pre-PR), instead of Abort.
        for (Value tensor : helperOperands) {
            if (isa<MemRefType>(tensor.getType()))
                continue;
            auto descOp = tensor.getDefiningOp<::tla::TensorDescOp>();
            if (!descOp || !llvm::isa<::tla::PtrType>(descOp.getBase().getType()))
                continue;
            Value ptr = descOp.getBase();
            auto itp = ptr.getDefiningOp<::tla::IntToPtrOp>();
            if (!itp)
                return rewriter.notifyMatchFailure(vecFuncOp, "expected pointer lowered to tla.inttoptr boundary");
            // Prefer storage-identity seed (AllocPtr / outside ptr_add): SCF +
            // UPDATE next_ptr chase. Fall back to the inttoptr addr only when
            // identity itself is still an address SSA.
            Value identity = getVectorHelperStorageIdentity(tensor);
            Value callAddr = identity;
            if (!dominanceInfo.dominates(callAddr, vecFuncOp.getOperation())) {
                callAddr = stripHelperAddressCasts(chaseScfCarrierInit(itp.getAddr()));
                // Peel in-region ptrtoint of a live next_ptr back toward seed.
                DenseSet<Value> peelVisited;
                while (peelVisited.insert(callAddr).second &&
                       !dominanceInfo.dominates(callAddr, vecFuncOp.getOperation())) {
                    if (auto p2i = callAddr.getDefiningOp<::tla::PtrToIntOp>()) {
                        callAddr = getVectorHelperStorageIdentity(p2i.getPtr());
                        continue;
                    }
                    if (auto p2i = callAddr.getDefiningOp<LLVM::PtrToIntOp>()) {
                        callAddr = getVectorHelperStorageIdentity(p2i.getArg());
                        continue;
                    }
                    break;
                }
            }
            if (!dominanceInfo.dominates(callAddr, vecFuncOp.getOperation()))
                return rewriter.notifyMatchFailure(
                    vecFuncOp,
                    "cannot outline address-backed tile whose base address is defined "
                    "inside tla.vec.func");
        }

        auto helperOr = buildHelperFunc(
            module, funcOp, vecFuncOp, helperOperands, helperOperandAliases, scalarOperands, nextVectorRegionId,
            loweredMemrefByValue);
        if (failed(helperOr))
            return vecFuncOp.emitOpError("failed to build vector helper function"), failure();
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
                // Helper call ABI: in-loop ``tensor_desc(inttoptr(scf_i64))``
                // materializes from the seed init (dominates the call). Live
                // iter_arg addresses are rebound inside the helper via
                // materializeTensorDescParameteri / resolveEffectiveUbPtr.
                if (auto itp = ptr.getDefiningOp<::tla::IntToPtrOp>()) {
                    Value liveAddr = stripHelperAddressCasts(itp.getAddr());
                    Value seedAddr = stripHelperAddressCasts(chaseScfCarrierInit(liveAddr));
                    // Once→for→once UnAlign: live addr may be in-region ptrtoint of
                    // next_ptr; peel to a dominating seed (same as outline check).
                    if (!dominanceInfo.dominates(seedAddr, vecFuncOp.getOperation())) {
                        Value identity = getVectorHelperStorageIdentity(tensor);
                        if (dominanceInfo.dominates(identity, vecFuncOp.getOperation())) {
                            OpBuilder::InsertionGuard guard(rewriter);
                            rewriter.setInsertionPoint(vecFuncOp);
                            if (identity.getType().isInteger(64) || isa<IndexType>(identity.getType()))
                                seedAddr = identity;
                            else if (isa<::tla::PtrType>(identity.getType()))
                                seedAddr = rewriter.create<::tla::PtrToIntOp>(
                                    vecFuncOp.getLoc(), rewriter.getI64Type(), identity);
                            else if (isa<LLVM::LLVMPointerType>(identity.getType()))
                                seedAddr = rewriter.create<LLVM::PtrToIntOp>(
                                    vecFuncOp.getLoc(), rewriter.getI64Type(), identity);
                        }
                    }
                    if (seedAddr != liveAddr) {
                        // Seed is the dominating i64 carrier init; rebuild inttoptr
                        // at the call site (addr SSA is never itself an inttoptr).
                        OpBuilder::InsertionGuard guard(rewriter);
                        rewriter.setInsertionPoint(vecFuncOp);
                        ptr =
                            rewriter.create<::tla::IntToPtrOp>(vecFuncOp.getLoc(), itp.getResult().getType(), seedAddr);
                    }
                }
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

        if (hasDebugPrint) {
            Value workspace = getOrAppendDebugPrintWorkspaceArg(funcOp);
            emitVfDebugBoundary(rewriter, vecFuncOp.getLoc(), module, workspace, true);
            module->setAttr("tla.vf_debug", UnitAttr::get(rewriter.getContext()));
        }
        auto call = rewriter.create<func::CallOp>(vecFuncOp.getLoc(), helper, callOperands);
        call->setAttr("hivm.vector_function", UnitAttr::get(rewriter.getContext()));
        call->setAttr("no_inline", UnitAttr::get(rewriter.getContext()));
        if (hasDebugPrint)
            emitVfDebugBoundary(rewriter, vecFuncOp.getLoc(), module, Value{}, false);
        rewriter.eraseOp(vecFuncOp);
        return success();
    }

private:
    ModuleOp module;
    int& nextVectorRegionId;
    DenseMap<Value, Value>& loweredMemrefByValue;
    bool& invalidScalarAccessBase;
};

} // namespace

void populateSimdVectorPatterns(
    RewritePatternSet& patterns, ModuleOp module, int& nextVectorRegionId, DenseMap<Value, Value>& loweredMemrefByValue,
    bool& invalidScalarAccessBase)
{
    MLIRContext* ctx = patterns.getContext();
    patterns.add<LowerVecFuncRegionPattern>(
        ctx, module, nextVectorRegionId, loweredMemrefByValue, invalidScalarAccessBase);
    populateTlaCopyPatterns(patterns, loweredMemrefByValue);
}

namespace {

// Neither vector pass has a "skip functions I own nothing in" gate, and that is
// not an oversight in either direction: no such gate can exist as the pipeline
// stands.
//
// mlir::applyPatternsGreedily does not only apply patterns -- it folds and CSEs
// everything it walks. So the *driver*, not the pattern set, performs part of the
// lowering, and "run the pass but match nothing" is not a no-op. Gating this pass
// on its own ops (vec.func(simd) / tla.copy / tla.vector) leaves 7 of the 82
// end-to-end examples textually different; tests/lit/scalar_ops/sqrt_kernel_body
// has no vector work at all and still differs, purely by constant order. There is
// no predicate that makes either pass skippable, because what they contribute to
// those functions is not vector lowering.
//
// Both passes therefore run unconditionally over every AIV/MIX function, which
// only composes if neither reorders the other's constant pool. Both disable
// constant CSE (GreedyRewriteConfig::cseConstants) and leave constant dedup to
// the pipeline's later CSEPass, which is where it belongs.
//
// Measured against the single combined pass, over all 82 end-to-end examples:
// one .o differs, by 2 bytes, in basic_mixed_store_zNUnAlign. With the SIMT
// driver no longer gated it folds literal-zero tile offsets before this pass
// outlines the vector region, so vector_region_0 is captured with 8 operands
// instead of 12 and four `add i64 0, x` identities disappear -- provably
// semantics-preserving, and smaller. 10 .ll and 135 .mlir differ by constant
// order and the SSA renumbering cascading from it.
//
// What matters for mode separation is that the *patterns* are mode-guarded, and
// they are.

/// Lower the SIMD half: a `tla.vec.func(mode="simd")` region maps onto
/// whole-vector AVE instructions. Also owns tla.copy lowering, which shares this
/// pass's base-memref cache, and the end-of-lowering cleanup that must run after
/// both halves.
///
/// Runs after tla-vector-region-simt.
class TlaVectorRegionSimdPass : public PassWrapper<TlaVectorRegionSimdPass, OperationPass<ModuleOp>> {
public:
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TlaVectorRegionSimdPass)

    StringRef getArgument() const override
    {
        return "tla-vector-region-simd";
    }
    StringRef getName() const override
    {
        return "TlaVectorRegionSimdPass";
    }
    StringRef getDescription() const override
    {
        return "Outline tla.vec.func(mode=simd) regions and lower fragment ops to vector IR.";
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

        // Extern signatures were checked in TLA types before ptr lowering.
        // Convert declarations here, e.g. !tla.vector<128xf16> to vector<128xf16>.
        for (func::FuncOp func : module.getOps<func::FuncOp>()) {
            if (!func.isDeclaration())
                continue;
            SmallVector<Type> inputs, results;
            for (Type type : func.getArgumentTypes()) {
                auto lowered = lowerSCFCarrierType(type);
                if (failed(lowered)) {
                    func.emitOpError("cannot lower extern argument type ") << type;
                    signalPassFailure();
                    return;
                }
                inputs.push_back(*lowered);
            }
            for (Type type : func.getResultTypes()) {
                auto lowered = lowerSCFCarrierType(type);
                if (failed(lowered)) {
                    func.emitOpError("cannot lower extern result type ") << type;
                    signalPassFailure();
                    return;
                }
                results.push_back(*lowered);
            }
            func.setType(FunctionType::get(&getContext(), inputs, results));
        }

        // Snapshot the functions up front: lowering a vec.func appends a new
        // vector_region helper to the module, and that helper must not be fed back
        // through the lowering/folding driver (it already holds lowered AVE ops and
        // the carried scf control flow).
        SmallVector<func::FuncOp, 4> funcOps(module.getOps<func::FuncOp>());
        for (func::FuncOp funcOp : funcOps) {
            if (funcOp.isDeclaration())
                continue;
            if (funcOp->hasAttr(kHivmVectorFunctionAttrName))
                continue;
            // The SIMT pass runs first and outlines its own helpers, which carry a
            // SIMT calling convention rather than hivm.vector_function. Skipping
            // them keeps this pass seeing exactly the functions the combined pass
            // saw, which snapshotted the module before any lowering happened.
            if (funcOp->hasAttr(hivm_regbaseintrins::kDavinciCallingConvAttrName))
                continue;
            std::optional<hivm::TFuncCoreType> coreType = getFunctionCoreType(funcOp.getOperation());
            if (coreType != hivm::TFuncCoreType::AIV && coreType != hivm::TFuncCoreType::MIX)
                continue;
            inlineVectorRegionWrappers(funcOp);
            // Fresh per-function lowering state: the base-memref handoff cache
            // shared by LowerCopyPattern (gm<->ub / ub->l1 cifax runtime calls) and
            // the vec.func helper operand materialization.
            ::tla::TlaTensorMemrefLowering lowering;
            RewritePatternSet patterns(&getContext());
            bool invalidScalarAccessBase = false;
            patterns.add<InlineVectorRegionWrapperPattern>(&getContext());
            populateSimdVectorPatterns(
                patterns, module, nextVectorRegionId, lowering.loweredMemrefByValue, invalidScalarAccessBase);
            GreedyRewriteConfig simdCfg;
            simdCfg.cseConstants = false;
            if (failed(mlir::applyPatternsGreedily(funcOp, std::move(patterns), simdCfg))) {
                signalPassFailure();
                return;
            }
            if (invalidScalarAccessBase) {
                signalPassFailure();
                return;
            }
        }

        // A pattern can decline a match without failing the greedy driver.
        // Never hand an unoutlined extern VF to the backend as successful IR.
        if (module
                .walk([&](::tla::CallExternOp call) {
                    call.emitOpError("failed to outline enclosing SIMD VF for extern @") << call.getCallee();
                    return WalkResult::interrupt();
                })
                .wasInterrupted()) {
            signalPassFailure();
            return;
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
        // remaining operations here as well so the vector lowering is the single
        // owner of tla.local_mem_bar lowering.
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
};

} // namespace

std::unique_ptr<Pass> createTlaVectorRegionSimdPass()
{
    return std::make_unique<TlaVectorRegionSimdPass>();
}

void registerTlaVectorRegionSimdPass()
{
    PassRegistration<TlaVectorRegionSimdPass>();
}

} // namespace tla
