#include "Dialect/Tla/IR/TlaOps.h"

#include <array>
#include <limits>
#include <numeric>
#include <optional>
#include <utility>

#include "Dialect/Tla/IR/TlaAttrs.h"
#include "Dialect/Tla/IR/TlaTypes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/OpImplementation.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Support/raw_ostream.h"

#define GET_OP_CLASSES
#include "tla/Ops.cpp.inc"

namespace tla {

static constexpr llvm::StringLiteral kPrintTensorSupportedDtypes = "f16, f32, i8, i16, i32, u8, u16, u32";
static constexpr unsigned kDebugPrintMaxFormatFields = 8;
static constexpr uint64_t kDebugPrintFifoBytes = 1024 * 1024;
static constexpr uint64_t kDebugPrintFormatTlvBytes = 24;
static constexpr uint64_t kDebugPrintFormatSlotBytes = 8;

static std::optional<uint64_t> getLocalPrintTensorByteOffset(
    const std::array<uint64_t, 4>& packedCoord, llvm::ArrayRef<int64_t> packedStride, uint64_t elementBytes)
{
    uint64_t elementOffset = 0;
    uint64_t maxElementOffset = std::numeric_limits<uint32_t>::max() / elementBytes;
    for (size_t index = 0; index < packedCoord.size(); ++index) {
        uint64_t stride = static_cast<uint64_t>(packedStride[index]);
        if (stride != 0 && packedCoord[index] > (maxElementOffset - elementOffset) / stride)
            return std::nullopt;
        elementOffset += packedCoord[index] * stride;
    }
    return elementOffset * elementBytes;
}

struct DebugPrintFormatInfo {
    unsigned fieldCount = 0;
    uint64_t generatedLength = 0;
};

template <typename TreeType>
static mlir::LogicalResult getIndexTreeLeavesForVerify(
    mlir::Operation* op, TreeType treeType, llvm::SmallVectorImpl<int64_t>& leaves, llvm::StringRef name)
{
    if (failed(getTlaIndexTreeLeaves(treeType.getTree(), leaves)))
        return op->emitOpError() << "failed to decode " << name;
    return mlir::success();
}

static bool isSupportedCmpElementType(mlir::Type elementType)
{
    if (elementType.isF16() || elementType.isF32())
        return true;
    auto intType = mlir::dyn_cast<mlir::IntegerType>(elementType);
    if (!intType)
        return false;
    unsigned width = intType.getWidth();
    // Frontend whitelist: u8/u16/u32 + i32. Signed/signless only at 32-bit;
    // unsigned also at 8/16 so VCmpU* can be selected for the top-half values.
    if (intType.isUnsigned())
        return width == 8 || width == 16 || width == 32;
    return (intType.isSignless() || intType.isSigned()) && width == 32;
}

static bool isSupportedPrintTensorInteger(mlir::Type elementType)
{
    auto integerType = mlir::dyn_cast<mlir::IntegerType>(elementType);
    if (!integerType)
        return false;
    unsigned width = integerType.getWidth();
    if (integerType.isSignless())
        return width == 8 || width == 16 || width == 32;
    return integerType.isUnsigned() && (width == 8 || width == 16 || width == 32);
}

static std::string typeToString(mlir::Type type)
{
    std::string text;
    llvm::raw_string_ostream os(text);
    type.print(os);
    return os.str();
}

static std::string printTensorDiagnosticTypeToken(mlir::Type type)
{
    auto integerType = mlir::dyn_cast<mlir::IntegerType>(type);
    if (integerType && integerType.isUnsigned())
        return "u" + std::to_string(integerType.getWidth());
    return typeToString(type);
}

static bool isSupportedDebugPrintScalar(mlir::Type type)
{
    return isSupportedPrintTensorInteger(type) || type.isF16() || type.isF32();
}

static mlir::LogicalResult scanDebugPrintFormat(mlir::Operation* op, llvm::StringRef format, DebugPrintFormatInfo& info)
{
    info = {};
    for (size_t i = 0; i < format.size();) {
        unsigned char c = static_cast<unsigned char>(format[i]);
        if (c == '\0')
            return op->emitOpError("format string must not contain embedded NUL");
        if (c > 0x7f)
            return op->emitOpError("format string must contain ASCII only");
        if (c == '{') {
            if (i + 1 >= format.size())
                return op->emitOpError("malformed format string");
            if (format[i + 1] == '{') {
                ++info.generatedLength;
                i += 2;
                continue;
            }
            if (format[i + 1] == '}') {
                ++info.fieldCount;
                if (info.fieldCount > kDebugPrintMaxFormatFields)
                    return op->emitOpError("formatted debug_print supports at most ")
                           << kDebugPrintMaxFormatFields << " operands";
                info.generatedLength += 2;
                i += 2;
                continue;
            }
            if (format.find('}', i + 1) == llvm::StringRef::npos)
                return op->emitOpError("malformed format string");
            return op->emitOpError("unsupported format field");
        }
        if (c == '}') {
            if (i + 1 < format.size() && format[i + 1] == '}') {
                ++info.generatedLength;
                i += 2;
                continue;
            }
            return op->emitOpError("malformed format string");
        }
        info.generatedLength += c == '%' ? 2 : 1;
        ++i;
    }
    return mlir::success();
}

static mlir::LogicalResult verifyDebugPrintRecordSize(mlir::Operation* op, const DebugPrintFormatInfo& info)
{
    uint64_t recordBytes =
        kDebugPrintFormatTlvBytes + info.fieldCount * kDebugPrintFormatSlotBytes + info.generatedLength + 1;
    recordBytes = (recordBytes + kDebugPrintFormatSlotBytes - 1) & ~(kDebugPrintFormatSlotBytes - 1);
    if (recordBytes > kDebugPrintFifoBytes)
        return op->emitOpError("formatted debug_print record exceeds ")
               << kDebugPrintFifoBytes << " byte debug FIFO limit";
    return mlir::success();
}

static mlir::LogicalResult parseDebugPrintFormat(mlir::Operation* op, llvm::StringRef format, unsigned& fieldCount)
{
    DebugPrintFormatInfo info;
    if (failed(scanDebugPrintFormat(op, format, info)))
        return mlir::failure();
    if (failed(verifyDebugPrintRecordSize(op, info)))
        return mlir::failure();
    fieldCount = info.fieldCount;
    return mlir::success();
}

static bool isSupportedCmpMode(llvm::StringRef mode)
{
    return llvm::StringSwitch<bool>(mode).Cases("lt", "le", "gt", "ge", true).Cases("eq", "ne", true).Default(false);
}

mlir::LogicalResult DynamicUbBaseOp::verify()
{
    auto resTy = llvm::dyn_cast<PtrType>(getResult().getType());
    if (!resTy)
        return emitOpError("result must be !tla.ptr");
    if (resTy.getAlignment() == 0)
        return emitOpError("result pointer alignment must be positive");
    // UB only: the launch-time declaration this region relies on is the UB/Data
    // Cache boundary, and no other memory has one.
    if (resTy.getAddrspace() != AddressSpace::ub)
        return emitOpError("dynamic_ub_base requires a ub !tla.ptr");
    // In bits, as alloc_ptr does: a packed fp4 buffer has an i4 pointee, which
    // has no whole-byte width but is still a fixed-width scalar. Asking for
    // bytes here rejected every sub-byte type, and said so with a message that
    // blamed the type for not being fixed-width.
    if (getBitSizeOfFixedWidthScalarType(resTy.getPointee()) <= 0)
        return emitOpError("dynamic_ub_base pointee must be a fixed-width scalar type");
    return mlir::success();
}

mlir::LogicalResult AllocPtrOp::verify()
{
    auto resTy = llvm::dyn_cast<PtrType>(getResult().getType());
    if (!resTy)
        return emitOpError("result must be !tla.ptr");
    if (resTy.getAlignment() == 0)
        return emitOpError("result pointer alignment must be positive");
    auto ms = resTy.getAddrspace();
    if (ms == AddressSpace::generic || ms == AddressSpace::gm)
        return emitOpError("alloc_ptr requires on-chip !tla.ptr (l1, l0a, l0b, l0c, ub)");
    int64_t sizeBytes = getSizeBytesAttr().getInt();
    if (sizeBytes <= 0)
        return emitOpError("size_bytes must be positive");
    // In bits: a packed fp4 buffer has an i4 pointee, which has no whole-byte
    // width but is still a fixed-width scalar. The allocation itself is always a
    // whole number of bytes, which the divisibility check below still enforces.
    int64_t elemBits = getBitSizeOfFixedWidthScalarType(resTy.getPointee());
    if (elemBits <= 0)
        return emitOpError("alloc_ptr pointee must be a fixed-width scalar type");
    if ((sizeBytes * 8) % elemBits != 0)
        return emitOpError("size_bytes must be a multiple of result pointee type size");
    return mlir::success();
}

mlir::LogicalResult TensorPtrOp::verify()
{
    auto resTy = mlir::dyn_cast<PtrType>(getPtr().getType());
    if (!resTy)
        return emitOpError("result must be !tla.ptr");
    if (auto tensorTy = mlir::dyn_cast<TlaTensorType>(getSrc().getType())) {
        if (tensorTy.getPtr() != resTy)
            return emitOpError("result ptr type must match the tensor's embedded pointer type");
    }
    return mlir::success();
}

mlir::LogicalResult PtrAddOp::verify()
{
    auto srcTy = mlir::dyn_cast<PtrType>(getPtr().getType());
    auto resTy = mlir::dyn_cast<PtrType>(getResult().getType());
    if (!srcTy || !resTy)
        return emitOpError("operands and result must be !tla.ptr");
    if (srcTy.getPointee() != resTy.getPointee())
        return emitOpError("result pointee type must match the source pointer's pointee");
    if (srcTy.getAddrspace() != resTy.getAddrspace())
        return emitOpError("result address space must match the source pointer's address space");
    return mlir::success();
}

mlir::LogicalResult TensorDescOp::verify()
{
    auto tensorType = getResult().getType();
    auto layout = tensorType.getLayout();
    auto layoutTag = layout.getLayoutTag();
    bool isLinear = isLinearLayout(layoutTag);
    bool isNZFamily = isNZFamilyLayout(layoutTag);
    if (!isLinear && !isNZFamily)
        return emitOpError("result must use a supported linear or NZFamily layout");

    llvm::SmallVector<int64_t, 4> shapeLeaves;
    llvm::SmallVector<int64_t, 4> strideLeaves;
    llvm::SmallVector<int64_t, 2> originLeaves;
    llvm::SmallVector<int64_t, 2> coordLeaves;
    if (!layout.getOrigin())
        return emitOpError("result layout must have an origin shape");
    if (failed(getIndexTreeLeavesForVerify(getOperation(), layout.getShape(), shapeLeaves, "layout shape")) ||
        failed(getIndexTreeLeavesForVerify(getOperation(), layout.getStride(), strideLeaves, "layout stride")) ||
        failed(getIndexTreeLeavesForVerify(getOperation(), layout.getOrigin(), originLeaves, "origin shape")) ||
        failed(getIndexTreeLeavesForVerify(getOperation(), tensorType.getCoord(), coordLeaves, "coordinate")))
        return mlir::failure();
    if (isLinear) {
        size_t rank = coordLeaves.size();
        if ((rank != 1 && rank != 2) || shapeLeaves.size() != rank || strideLeaves.size() != rank ||
            originLeaves.size() != rank)
            return emitOpError("linear result metadata must have a consistent raw rank of one or two");
    }
    if (isNZFamily &&
        (shapeLeaves.size() != 4 || strideLeaves.size() != 4 || originLeaves.size() != 2 || coordLeaves.size() != 2))
        return emitOpError("NZFamily layout shape/stride must have four leaves and origin/coordinate two leaves");

    if (isLinear) {
        auto isConstantOne = [](mlir::Value value) {
            llvm::APInt constant;
            return mlir::matchPattern(value, mlir::m_ConstantInt(&constant)) && constant == 1;
        };
        if (!isConstantOne(getShape2()) || !isConstantOne(getShape3()) || !isConstantOne(getStride2()) ||
            !isConstantOne(getStride3()))
            return emitOpError("linear layout shape[2:4] and stride[2:4] must be constant 1");
    }
    return mlir::success();
}

// Walk the enclosing ops looking for an ancestor of type AncestorOp. The
// required region may be several levels up (e.g. a compute op nested inside a
// scf.for loop inside a tla.vec.func), so this checks all transitive parents
// rather than just the immediate one.
template <typename AncestorOp>
static bool hasEnclosing(mlir::Operation* op)
{
    for (mlir::Operation* parent = op->getParentOp(); parent; parent = parent->getParentOp())
        if (mlir::isa<AncestorOp>(parent))
            return true;
    return false;
}

// The region-wrapper requirement is a frontend/authoring constraint, enforced
// while ops still live in the tla.func container. Once tla-lower-func lowers
// tla.func to func.func -- and the tla.vector lowering passes /
// TlaSplitMixedFuncPass
// inline the frontend tla.cube / tla.vector wrappers into the resulting
// func.func (carrying the AIC/AIV/MIX core context on function attributes such
// as hivm.func_core_type / hivm.part_of_mix / hacc.entry rather than a lexical
// region) -- the lexical wrapper is legitimately gone. Ops already inside a
// lowered func.func are therefore exempt; the constraint is fully enforced at
// parse time and in the frontend, where ops are still under tla.func.
static bool isInLoweredFunc(mlir::Operation* op)
{
    return op->getParentOfType<mlir::func::FuncOp>() != nullptr;
}

template <typename AncestorOp>
static bool hasEnclosingRegion(mlir::Operation* op)
{
    return hasEnclosing<AncestorOp>(op) || isInLoweredFunc(op);
}

mlir::LogicalResult MmadOp::verify()
{
    if (!hasEnclosingRegion<CubeOp>(getOperation()))
        return emitOpError("must be nested inside a tla.cube region");

    // HF32 rounding only applies to f32 L0A/L0B operands; requesting it for any
    // other source element type (f16/bf16/...) is meaningless and must be rejected.
    HF32Mode mode = getHf32Mode().getValue();
    if (mode != HF32Mode::HF32_DISABLE) {
        auto checkF32SourceOperand = [&](TlaTensorType operandType,
                                         llvm::StringRef operandName) -> mlir::LogicalResult {
            mlir::Type elementType = operandType.getPtr().getPointee();
            if (!elementType.isF32())
                return emitOpError() << "hf32_mode " << stringifyHF32Mode(mode) << " requires f32 source operands, but "
                                     << operandName << " operand has element type " << elementType;
            return mlir::success();
        };

        if (failed(checkF32SourceOperand(getLhs().getType(), "lhs")))
            return mlir::failure();
        if (failed(checkF32SourceOperand(getRhs().getType(), "rhs")))
            return mlir::failure();
    }

    return mlir::success();
}

mlir::LogicalResult CallExternOp::verify()
{
    auto vf = (*this)->getParentOfType<VecFuncOp>();
    bool inVector = hasEnclosing<VectorOp>(getOperation());
    bool inCube = hasEnclosing<CubeOp>(getOperation());
    if (vf) {
        // The enclosing vector wrapper is gone between SIMT and SIMD lowering.
        if (inCube || !hasEnclosingRegion<VectorOp>(getOperation()))
            return emitOpError("SIMD extern requires a vector region");
        auto mode = vf->getAttrOfType<mlir::StringAttr>("mode");
        if (mode && mode.getValue().equals_insensitive("simt"))
            return emitOpError("not supported inside a SIMT tla.vec.func");
    } else if (inVector == inCube)
        return emitOpError("must be nested inside exactly one tla.vector or tla.cube region");
    auto isExternRegister = [](mlir::Type type) {
        auto reg = mlir::dyn_cast<VectorSSAType>(type);
        if (!reg)
            return false;
        auto element = reg.getElementType();
        auto integer = mlir::dyn_cast<mlir::IntegerType>(element);
        bool supported = element.isF16() || element.isBF16() || element.isF32() ||
                         (integer && !integer.isSigned() &&
                          (integer.getWidth() == 8 || integer.getWidth() == 16 || integer.getWidth() == 32 ||
                           integer.getWidth() == 64));
        return supported && reg.getValidLanes() == 256 * 8 / element.getIntOrFloatBitWidth();
    };
    for (mlir::Value operand : getOperands()) {
        mlir::Type type = operand.getType();
        if (mlir::isa<VectorSSAType>(type)) {
            if (!vf || !isExternRegister(type))
                return emitOpError(
                    "register arguments require a SIMD VF and a full 256-byte register "
                    "of f16, bf16, f32, or 8/16/32/64-bit integer elements");
            if (!vf.getBody().isAncestor(operand.getParentRegion()))
                return emitOpError("register arguments must be defined in the same SIMD VF");
            continue;
        }
        if (mlir::isa<PtrType, mlir::IntegerType, mlir::FloatType>(type))
            continue;
        return emitOpError(
                   "operands must be !tla.ptr, scalar integer/float (not index), or a supported !tla.vector, got ")
               << type;
    }
    if (getNumResults() > 1)
        return emitOpError("supports at most one register result");
    for (mlir::Value result : getResults()) {
        if (!vf || !isExternRegister(result.getType()))
            return emitOpError(
                "register results require a SIMD VF and a full 256-byte register "
                "of f16, bf16, f32, or 8/16/32/64-bit integer elements");
        for (mlir::Operation* user : result.getUsers())
            if (user->getParentOfType<VecFuncOp>() != vf)
                return emitOpError("register results must be used in the same SIMD VF");
    }
    return mlir::success();
}

mlir::LogicalResult MmadMxOp::verify()
{
    if (!hasEnclosingRegion<CubeOp>(getOperation()))
        return emitOpError("must be nested inside a tla.cube region");

    auto accTy = mlir::dyn_cast<TlaTensorType>(getAcc().getType());
    auto lhsTy = mlir::dyn_cast<TlaTensorType>(getLhs().getType());
    auto rhsTy = mlir::dyn_cast<TlaTensorType>(getRhs().getType());
    if (!accTy || !lhsTy || !rhsTy)
        return mlir::success(); // Operand type verifier handles malformed tensors.

    if (accTy.getPtr().getAddrspace() != AddressSpace::l0c)
        return emitOpError("acc must be an l0c tile");
    if (lhsTy.getPtr().getAddrspace() != AddressSpace::l0a)
        return emitOpError("lhs must be an l0a tile");
    if (rhsTy.getPtr().getAddrspace() != AddressSpace::l0b)
        return emitOpError("rhs must be an l0b tile");
    // Every MX route accumulates into fp32; there is no narrowing mad_mx form.
    if (!accTy.getPtr().getPointee().isF32())
        return emitOpError("acc must be f32: every MX route accumulates into an fp32 L0C");

    LayoutTag lhsTag = lhsTy.getLayout().getLayoutTag();
    LayoutTag rhsTag = rhsTy.getLayout().getLayoutTag();
    if (accTy.getLayout().getLayoutTag() != LayoutTag::L0Clayout)
        return emitOpError("acc must be tagged L0Clayout");

    // Two operand forms, and the sides must agree on which one they are in:
    // packed fp4 (!tla.f4e2m1 / !tla.f4e1m2, which mix freely) or fp8
    // (f8E4M3FN / f8E5M2, which also mix freely), both in zN / nZ.
    bool lhsFp4 = ::tla::isPackedFp4Type(lhsTy.getPtr().getPointee());
    bool rhsFp4 = ::tla::isPackedFp4Type(rhsTy.getPtr().getPointee());
    if (lhsFp4 != rhsFp4)
        return emitOpError("both operands must be packed fp4 or neither");

    auto isFp8 = [](mlir::Type ty) { return ty.isFloat8E4M3FN() || ty.isFloat8E5M2(); };
    if (lhsFp4) {
        if (lhsTag != LayoutTag::zN || rhsTag != LayoutTag::nZ)
            return emitOpError("packed fp4 operands must be tagged zN / nZ");
    } else {
        if (lhsTag != LayoutTag::zN || rhsTag != LayoutTag::nZ)
            return emitOpError("fp8 MX operands must be tagged zN (lhs) and nZ (rhs)");
        if (!isFp8(lhsTy.getPtr().getPointee()) || !isFp8(rhsTy.getPtr().getPointee()))
            return emitOpError("MX operands must be f8E4M3FN / f8E5M2, or fp4-tagged i8 tiles");
    }
    return mlir::success();
}

mlir::LogicalResult CopyMxOp::verify()
{
    if (!hasEnclosingRegion<CubeOp>(getOperation()))
        return emitOpError("must be nested inside a tla.cube region");
    auto srcTy = mlir::dyn_cast<TlaTensorType>(getSrc().getType());
    auto dstTy = mlir::dyn_cast<TlaTensorType>(getDst().getType());
    auto scaleTy = mlir::dyn_cast<TlaTensorType>(getScale().getType());
    if (!srcTy || !dstTy || !scaleTy)
        return mlir::success(); // Operand type verifier handles malformed tensors.

    AddressSpace src = srcTy.getPtr().getAddrspace();
    AddressSpace dst = dstTy.getPtr().getAddrspace();
    AddressSpace scale = scaleTy.getPtr().getAddrspace();
    if (src != AddressSpace::l1)
        return emitOpError("MX operand tile must come from l1");
    if (dst != AddressSpace::l0a && dst != AddressSpace::l0b)
        return emitOpError("MX operand tile must land in l0a or l0b");
    if (scale != AddressSpace::l1)
        return emitOpError("MX scale tile must live in l1");

    // The scale block is e8m0, end to end: the frontend binds GM scale buffers
    // through from_dlpack(Float8E8M0) and the L1 tile is allocated as
    // Float8E8M0 too. Only the materialized memref stores bytes (i8), which is
    // the storageElementType boundary, not a spelling callers should rely on.
    auto scaleElem = scaleTy.getPtr().getPointee();
    if (!::llvm::isa<::tla::Float8E8M0Type>(scaleElem))
        return emitOpError("MX scale tile must be f8e8m0");
    // A-side scales feed L0A and must be zZMxScale; B-side feed L0B and must be nNMxScale.
    LayoutTag expected = dst == AddressSpace::l0a ? LayoutTag::zZMxScale : LayoutTag::nNMxScale;
    if (scaleTy.getLayout().getLayoutTag() != expected)
        return emitOpError("MX scale tile for an ")
               << (dst == AddressSpace::l0a ? "l0a" : "l0b") << " destination must be tagged "
               << (dst == AddressSpace::l0a ? "zZMxScale" : "nNMxScale");
    return mlir::success();
}

mlir::LogicalResult Img2colOp::verify()
{
    if (!hasEnclosingRegion<CubeOp>(getOperation()))
        return emitOpError("must be nested inside a tla.cube region");
    auto srcTy = mlir::dyn_cast<TlaTensorType>(getSrc().getType());
    auto dstTy = mlir::dyn_cast<TlaTensorType>(getDst().getType());
    if (!srcTy || !dstTy)
        return mlir::success(); // Operand type verifier handles malformed tensors.

    if (srcTy.getPtr().getAddrspace() != AddressSpace::l1)
        return emitOpError("img2col source must be an l1 feature-map tile");
    if (srcTy.getLayout().getLayoutTag() != LayoutTag::zNUnAlign)
        return emitOpError("img2col source must be tagged zNUnAlign");
    AddressSpace dst = dstTy.getPtr().getAddrspace();
    if (dst != AddressSpace::l0a)
        return emitOpError("img2col destination must be an l0a tile");
    if (dstTy.getLayout().getLayoutTag() != LayoutTag::zN)
        return emitOpError("img2col destination must be tagged zN");
    if (srcTy.getPtr().getPointee() != dstTy.getPtr().getPointee())
        return emitOpError("img2col source and destination element types must match");
    // The A5 img2col path is instantiated for 2- and 4-byte elements only
    // (f16/bf16/f32); int8/fp8/fp4 have no img2col route, so refuse them here
    // rather than at lowering time.
    ::mlir::Type elem = srcTy.getPtr().getPointee();
    if (!elem.isF16() && !elem.isBF16() && !elem.isF32())
        return emitOpError("img2col element type must be f16, bf16 or f32, but got ") << elem;

    // Validate the packing contract before lowering narrows hardware fields.
    // A dynamic filter or K window cannot establish whole-C1 boundaries, so
    // reject it rather than silently configuring a different convolution.
    int64_t filterExtents[2];
    unsigned filterAxis = 0;
    for (mlir::Value extent : {getFilterH(), getFilterW()}) {
        llvm::APInt value;
        if (!mlir::matchPattern(extent, mlir::m_ConstantInt(&value)))
            return emitOpError("filter_hw must be compile-time constants for K-window validation");
        int64_t size = value.getSExtValue();
        if (size < 1 || size > 511)
            return emitOpError("filter_hw entries must be in 1..511");
        filterExtents[filterAxis++] = size;
    }

    // These fields are narrowed to uint16_t/uint8_t in the device helper.
    // Reject out-of-range constants here; dynamic values retain the documented
    // runtime precondition because their value is not available to the verifier.
    auto verifyConstantRange = [&](mlir::Value value, llvm::StringRef name, int64_t min,
                                   int64_t max) -> mlir::LogicalResult {
        llvm::APInt constant;
        if (!mlir::matchPattern(value, mlir::m_ConstantInt(&constant)))
            return mlir::success();
        int64_t integer = constant.getSExtValue();
        if (integer < min || integer > max)
            return emitOpError() << name << " must be in " << min << ".." << max << ", got " << integer;
        return mlir::success();
    };
    for (auto [value, name] : {std::pair{getFmapH(), "fmap_h"}, std::pair{getFmapW(), "fmap_w"}}) {
        if (mlir::failed(verifyConstantRange(value, name, 1, 65535)))
            return mlir::failure();
    }
    for (auto [value, name] :
         {std::pair{getStrideH(), "stride_h"}, std::pair{getStrideW(), "stride_w"},
          std::pair{getDilationH(), "dilation_h"}, std::pair{getDilationW(), "dilation_w"}}) {
        if (mlir::failed(verifyConstantRange(value, name, 1, 255)))
            return mlir::failure();
    }
    for (auto [value, name] :
         {std::pair{getPadLeft(), "pad_left"}, std::pair{getPadRight(), "pad_right"}, std::pair{getPadTop(), "pad_top"},
          std::pair{getPadBottom(), "pad_bottom"}}) {
        if (mlir::failed(verifyConstantRange(value, name, 0, 255)))
            return mlir::failure();
    }

    llvm::SmallVector<int64_t> srcShape, dstShape, dstCoord;
    if (failed(getTlaIndexTreeLeaves(srcTy.getLayout().getShape().getTree(), srcShape)) ||
        failed(getTlaIndexTreeLeaves(dstTy.getLayout().getShape().getTree(), dstShape)) ||
        failed(getTlaIndexTreeLeaves(dstTy.getCoord().getTree(), dstCoord)) || srcShape.size() != 4 ||
        dstShape.size() != 4 || dstCoord.size() != 2)
        return emitOpError("requires four-leaf source/destination shapes and a two-axis destination coordinate");
    int64_t c0 = elem.isF32() ? 8 : 16;
    if (srcShape[2] != c0 || srcShape[3] < 1 || srcShape[3] > 65535 / c0)
        return emitOpError("source channels must be static whole C1 slices within 1..65535 channels");
    int64_t channels = c0 * srcShape[3];
    if (dstShape[2] != c0 || dstShape[3] < 1 || dstShape[3] > 65535 / c0)
        return emitOpError("destination K extent must be static, C0-aligned and within 1..65535");
    int64_t kExtent = c0 * dstShape[3];
    int64_t sliceK = filterExtents[0] * filterExtents[1] * c0;
    if (kExtent % sliceK != 0)
        return emitOpError("destination K extent must span whole C1 slices (filter_h * filter_w * C0)");
    if (dstCoord[1] < 0 || dstCoord[1] > 65535 || dstCoord[1] % sliceK != 0)
        return emitOpError("destination K coordinate must be static, within 0..65535 and on a whole C1 slice");
    int64_t fullK = channels * filterExtents[0] * filterExtents[1];
    if (dstCoord[1] + kExtent > fullK)
        return emitOpError("destination K window exceeds the source channel/filter extent");
    return mlir::success();
}

mlir::LogicalResult FillOp::verify()
{
    if (!hasEnclosingRegion<CubeOp>(getOperation()))
        return emitOpError("must be nested inside a tla.cube region");
    auto dstTy = mlir::dyn_cast<TlaTensorType>(getDst().getType());
    if (!dstTy)
        return mlir::success(); // Operand type verifier handles malformed tensors.

    if (dstTy.getPtr().getAddrspace() != AddressSpace::l1)
        return emitOpError("fill destination must be an l1 tile");
    LayoutTag tag = dstTy.getLayout().getLayoutTag();
    if (tag != LayoutTag::zN && tag != LayoutTag::nZ)
        return emitOpError("fill destination must be tagged zN or nZ");
    ::mlir::Type elem = dstTy.getPtr().getPointee();
    bool zeroOnly = ::tla::isPackedFp4Type(elem) || mlir::isa<mlir::Float8E4M3FNType, mlir::Float8E5M2Type>(elem);
    if (!zeroOnly && !::tla::isOfficialL1FillElementType(elem))
        return emitOpError("unsupported L1 fill destination element type");
    if (zeroOnly) {
        ::llvm::APInt valueBits;
        if (!mlir::matchPattern(getValue(), mlir::m_ConstantInt(&valueBits)) || !valueBits.isZero())
            return emitOpError("fp4/fp8 fill requires a constant zero bit pattern");
    }
    return mlir::success();
}
// One thread block may hold at most this many threads on the supported targets.
// The lowering packs the product into hivm_regbaseintrins::SIMT_EntryAttr, whose
// value is a uint32_t, so an unchecked product would also truncate.
static constexpr int64_t kMaxSimtThreadsPerBlock = 2048;

mlir::LogicalResult VecFuncOp::verify()
{
    if (!hasEnclosingRegion<VectorOp>(getOperation()))
        return emitOpError("must be nested inside a tla.vector region");

    mlir::Operation* op = getOperation();
    // A missing mode means "simd" -- that is the documented default and what the
    // lowering assumes -- but a mode that is spelled out must be one we know,
    // otherwise a typo silently lowers as SIMD.
    bool isSimt = false;
    if (auto modeAttr = op->getAttrOfType<mlir::StringAttr>("mode")) {
        llvm::StringRef mode = modeAttr.getValue();
        isSimt = mode.equals_insensitive("simt");
        if (!isSimt && !mode.equals_insensitive("simd"))
            return emitOpError() << "mode must be \"simd\" or \"simt\", got \"" << mode << "\"";
    }

    auto dimsAttr = op->getAttrOfType<mlir::DenseI64ArrayAttr>("thread_block_dim");
    if (!isSimt) {
        if (dimsAttr)
            return emitOpError("'thread_block_dim' is only valid with mode = \"simt\"");
        return mlir::success();
    }

    if (!dimsAttr)
        return emitOpError("mode = \"simt\" requires a 'thread_block_dim' attribute");
    if (dimsAttr.size() != 3)
        return emitOpError() << "'thread_block_dim' must have exactly 3 elements, got " << dimsAttr.size();

    int64_t product = 1;
    for (int i = 0; i < 3; ++i) {
        int64_t extent = dimsAttr[i];
        if (extent < 1)
            return emitOpError() << "'thread_block_dim' entries must be positive, got " << extent << " at index " << i;
        product *= extent;
    }
    if (product > kMaxSimtThreadsPerBlock)
        return emitOpError() << "thread_block_dim describes " << product
                             << " threads per block, more than the supported maximum of " << kMaxSimtThreadsPerBlock;

    return mlir::success();
}

static mlir::LogicalResult verifyInterleaveLikeElementTypes(
    mlir::Operation* op, VectorSSAType src0Type, VectorSSAType src1Type, VectorSSAType dst0Type, VectorSSAType dst1Type)
{
    mlir::Type src0ElementType = src0Type.getElementType();
    mlir::Type src1ElementType = src1Type.getElementType();
    mlir::Type dst0ElementType = dst0Type.getElementType();
    mlir::Type dst1ElementType = dst1Type.getElementType();
    if (src0ElementType != src1ElementType || src0ElementType != dst0ElementType || src0ElementType != dst1ElementType)
        return op->emitOpError() << "requires all operands and results to have the same element "
                                    "type, got "
                                 << "src0=" << src0ElementType << ", src1=" << src1ElementType
                                 << ", dst0=" << dst0ElementType << ", dst1=" << dst1ElementType;

    return mlir::success();
}

static mlir::LogicalResult verifyInterleaveLikeMaskTypes(
    mlir::Operation* op, MaskSSAType src0Type, MaskSSAType src1Type, MaskSSAType dst0Type, MaskSSAType dst1Type)
{
    if (src1Type != src0Type || dst0Type != src0Type || dst1Type != src0Type)
        return op->emitOpError("requires all operands and results to have the same MaskSSA type");

    int64_t lanes = src0Type.getPhysicalLanes();
    if (lanes != 64 && lanes != 128 && lanes != 256)
        return op->emitOpError("supports only mask<64>, mask<128>, and mask<256>; mask<32>/b64 is unsupported");

    return mlir::success();
}

mlir::LogicalResult PackOp::verify()
{
    if (!hasEnclosingRegion<VecFuncOp>(getOperation()))
        return emitOpError("must be nested inside a tla.vec.func region");

    if (auto src = mlir::dyn_cast<VectorSSAType>(getSrc().getType())) {
        auto dst = mlir::dyn_cast<VectorSSAType>(getResult().getType());
        if (!dst)
            return emitOpError("requires source and result to both be VectorSSA or both be MaskSSA");
        if (auto region = getOperation()->getParentOfType<VecFuncOp>())
            if (auto mode = region->getAttrOfType<mlir::StringAttr>("mode"))
                if (!mode.getValue().equals_insensitive("simd"))
                    return emitOpError("is only supported in tla.vec.func(mode='simd')");

        // 64-bit sources are not supported: Ascend 950 has no native
        // vpack for that width.
        auto srcInt = mlir::dyn_cast<mlir::IntegerType>(src.getElementType());
        auto dstInt = mlir::dyn_cast<mlir::IntegerType>(dst.getElementType());
        if (!srcInt || !dstInt || !dstInt.isUnsigned() || (dstInt.getWidth() != 8 && dstInt.getWidth() != 16) ||
            srcInt.getWidth() != 2 * dstInt.getWidth())
            return emitOpError() << "unsupported vector element type pair " << src.getElementType() << " -> "
                                 << dst.getElementType()
                                 << "; expected 16->8 or 32->16 integers with an unsigned result";

        // VectorSSAType caps values at one 256-byte register (RegTraitNumOne).
        // CANN RegTraitNumTwo supports LOW only, but is not represented here.
        if (src.getValidLanes() != 2048 / srcInt.getWidth() || dst.getValidLanes() != 2048 / dstInt.getWidth())
            return emitOpError("requires full 256-byte vector registers (RegTraitNumOne)");
        return mlir::success();
    }
    if (auto src = mlir::dyn_cast<MaskSSAType>(getSrc().getType())) {
        auto dst = mlir::dyn_cast<MaskSSAType>(getResult().getType());
        if (!dst)
            return emitOpError("requires source and result to both be VectorSSA or both be MaskSSA");
        int64_t srcLanes = src.getPhysicalLanes();
        int64_t resultLanes = dst.getPhysicalLanes();
        if (srcLanes != 64 && srcLanes != 128)
            return emitOpError() << "unsupported mask source width " << srcLanes << "; expected 64 or 128 lanes";
        if (resultLanes != 2 * srcLanes)
            return emitOpError() << "mask result must have twice the source lane count, got source " << srcLanes
                                 << " and result " << resultLanes;
        return mlir::success();
    }
    return emitOpError("expects VectorSSA or MaskSSA operands");
}

mlir::LogicalResult UnpackOp::verify()
{
    if (!hasEnclosingRegion<VecFuncOp>(getOperation()))
        return emitOpError("must be nested inside a tla.vec.func region");
    if (auto region = getOperation()->getParentOfType<VecFuncOp>())
        if (auto mode = region->getAttrOfType<mlir::StringAttr>("mode"))
            if (!mode.getValue().equals_insensitive("simd"))
                return emitOpError("is only supported in tla.vec.func(mode='simd')");

    auto src = mlir::cast<VectorSSAType>(getSrc().getType());
    auto dst = mlir::cast<VectorSSAType>(getResult().getType());
    // 64-bit destinations are not supported: Ascend 950 has no native
    // vunpack for that width.
    auto srcInt = mlir::dyn_cast<mlir::IntegerType>(src.getElementType());
    auto dstInt = mlir::dyn_cast<mlir::IntegerType>(dst.getElementType());
    if (!srcInt || !dstInt || srcInt.isUnsigned() != dstInt.isUnsigned() ||
        (srcInt.getWidth() != 8 && srcInt.getWidth() != 16) || dstInt.getWidth() != 2 * srcInt.getWidth())
        return emitOpError() << "unsupported vector element type pair " << src.getElementType() << " -> "
                             << dst.getElementType() << "; expected 8->16 or 16->32 integers with matching signedness";

    // VectorSSAType caps values at one 256-byte register (RegTraitNumOne).
    // CANN RegTraitNumTwo supports LOW only, but is not represented here.
    if (src.getValidLanes() != 2048 / srcInt.getWidth() || dst.getValidLanes() != 2048 / dstInt.getWidth())
        return emitOpError("requires full 256-byte vector registers (RegTraitNumOne)");
    return mlir::success();
}

mlir::LogicalResult InterleaveOp::verify()
{
    if (!hasEnclosingRegion<VecFuncOp>(getOperation()))
        return emitOpError("must be nested inside a tla.vec.func region");

    if (auto srcType = mlir::dyn_cast<VectorSSAType>(getSrc0().getType())) {
        auto src1Type = mlir::dyn_cast<VectorSSAType>(getSrc1().getType());
        auto dst0Type = mlir::dyn_cast<VectorSSAType>(getDst0().getType());
        auto dst1Type = mlir::dyn_cast<VectorSSAType>(getDst1().getType());
        if (!src1Type || !dst0Type || !dst1Type)
            return emitOpError("requires all operands and results to be VectorSSA or all to be MaskSSA");
        return verifyInterleaveLikeElementTypes(getOperation(), srcType, src1Type, dst0Type, dst1Type);
    }
    if (auto srcType = mlir::dyn_cast<MaskSSAType>(getSrc0().getType())) {
        auto src1Type = mlir::dyn_cast<MaskSSAType>(getSrc1().getType());
        auto dst0Type = mlir::dyn_cast<MaskSSAType>(getDst0().getType());
        auto dst1Type = mlir::dyn_cast<MaskSSAType>(getDst1().getType());
        if (!src1Type || !dst0Type || !dst1Type)
            return emitOpError("requires all operands and results to be VectorSSA or all to be MaskSSA");
        return verifyInterleaveLikeMaskTypes(getOperation(), srcType, src1Type, dst0Type, dst1Type);
    }
    return emitOpError("expects VectorSSA or MaskSSA operands");
}

mlir::LogicalResult DeInterleaveOp::verify()
{
    if (!hasEnclosingRegion<VecFuncOp>(getOperation()))
        return emitOpError("must be nested inside a tla.vec.func region");

    if (auto srcType = mlir::dyn_cast<VectorSSAType>(getSrc0().getType())) {
        auto src1Type = mlir::dyn_cast<VectorSSAType>(getSrc1().getType());
        auto dst0Type = mlir::dyn_cast<VectorSSAType>(getDst0().getType());
        auto dst1Type = mlir::dyn_cast<VectorSSAType>(getDst1().getType());
        if (!src1Type || !dst0Type || !dst1Type)
            return emitOpError("requires all operands and results to be VectorSSA or all to be MaskSSA");
        return verifyInterleaveLikeElementTypes(getOperation(), srcType, src1Type, dst0Type, dst1Type);
    }
    if (auto srcType = mlir::dyn_cast<MaskSSAType>(getSrc0().getType())) {
        auto src1Type = mlir::dyn_cast<MaskSSAType>(getSrc1().getType());
        auto dst0Type = mlir::dyn_cast<MaskSSAType>(getDst0().getType());
        auto dst1Type = mlir::dyn_cast<MaskSSAType>(getDst1().getType());
        if (!src1Type || !dst0Type || !dst1Type)
            return emitOpError("requires all operands and results to be VectorSSA or all to be MaskSSA");
        return verifyInterleaveLikeMaskTypes(getOperation(), srcType, src1Type, dst0Type, dst1Type);
    }
    return emitOpError("expects VectorSSA or MaskSSA operands");
}

// Register predicates describe the physical lane width of a 256-byte data
// register. Valid data lanes may be smaller, but predicate compatibility is
// determined by element width rather than VectorSSA valid_lanes.
static mlir::FailureOr<int64_t> getPhysicalVectorLanes(VectorSSAType vectorType)
{
    constexpr int64_t kVectorRegisterBytes = 256;
    int64_t elementBytes = getByteSizeOfFixedWidthScalarType(vectorType.getElementType());
    if (elementBytes <= 0 || kVectorRegisterBytes % elementBytes != 0)
        return mlir::failure();
    return kVectorRegisterBytes / elementBytes;
}

static mlir::LogicalResult verifyMaskMatchesVector(mlir::Operation* op, mlir::Value mask, VectorSSAType vectorType)
{
    if (!mask)
        return mlir::success();
    auto maskType = mlir::dyn_cast<MaskSSAType>(mask.getType());
    if (!maskType)
        return op->emitOpError("expected a !tla.mask<N> predicate");
    auto expectedPhysicalLanes = getPhysicalVectorLanes(vectorType);
    if (mlir::failed(expectedPhysicalLanes))
        return op->emitOpError() << "cannot derive predicate lanes for " << vectorType;
    if (maskType.getPhysicalLanes() != *expectedPhysicalLanes)
        return op->emitOpError() << "mask has " << maskType.getPhysicalLanes() << " predicate lanes, expected "
                                 << *expectedPhysicalLanes << " for " << vectorType.getElementType() << " VectorSSA";
    return mlir::success();
}

// Vector compute ops must live inside a tla.vec.func region.
#define TLA_VERIFY_IN_VEC_FUNC(OpTy)                                           \
    mlir::LogicalResult OpTy::verify()                                         \
    {                                                                          \
        if (!hasEnclosingRegion<VecFuncOp>(getOperation()))                    \
            return emitOpError("must be nested inside a tla.vec.func region"); \
        return mlir::success();                                                \
    }

// tla.full broadcasts a scalar, or a *one-lane* vector fragment such as a
// tla.reduce result. A full-width VectorSSA source is rejected: the lowering
// would silently splat lane 0 and drop the rest. The optional predicate must
// match the result, checked with the same helper the binary ops use.
mlir::LogicalResult FullOp::verify()
{
    if (!hasEnclosingRegion<VecFuncOp>(getOperation()))
        return emitOpError("must be nested inside a tla.vec.func region");
    auto resultType = mlir::dyn_cast<VectorSSAType>(getResult().getType());
    if (!resultType)
        return emitOpError("expected a !tla.vector result");
    if (auto srcType = mlir::dyn_cast<VectorSSAType>(getValue().getType())) {
        auto lanes = srcType.getValidLanes();
        if (lanes != 1)
            return emitOpError() << "vector source must be a one-lane fragment, got " << lanes
                                 << " valid lanes; broadcasting a full-width vector would keep "
                                    "lane 0 only";
    }
    return verifyMaskMatchesVector(getOperation(), getMask(), resultType);
}

#undef TLA_VERIFY_IN_VEC_FUNC

#define TLA_VERIFY_MASKED_VECTOR_LHS(OpTy)                                             \
    mlir::LogicalResult OpTy::verify()                                                 \
    {                                                                                  \
        if (!hasEnclosingRegion<VecFuncOp>(getOperation()))                            \
            return emitOpError("must be nested inside a tla.vec.func region");         \
        return verifyMaskMatchesVector(getOperation(), getMask(), getLhs().getType()); \
    }

TLA_VERIFY_MASKED_VECTOR_LHS(AddOp)
TLA_VERIFY_MASKED_VECTOR_LHS(SubOp)
TLA_VERIFY_MASKED_VECTOR_LHS(MulOp)
TLA_VERIFY_MASKED_VECTOR_LHS(DivOp)
TLA_VERIFY_MASKED_VECTOR_LHS(MaxOp)
TLA_VERIFY_MASKED_VECTOR_LHS(MinOp)
TLA_VERIFY_MASKED_VECTOR_LHS(AddsOp)
TLA_VERIFY_MASKED_VECTOR_LHS(SubsOp)
TLA_VERIFY_MASKED_VECTOR_LHS(MulsOp)
TLA_VERIFY_MASKED_VECTOR_LHS(DivsOp)
TLA_VERIFY_MASKED_VECTOR_LHS(MaxsOp)
TLA_VERIFY_MASKED_VECTOR_LHS(MinsOp)

#undef TLA_VERIFY_MASKED_VECTOR_LHS

static mlir::LogicalResult verifyShiftSourceAndResult(
    mlir::Operation* op, VectorSSAType sourceType, VectorSSAType resultType, mlir::Value mask)
{
    auto sourceElement = mlir::dyn_cast<mlir::IntegerType>(sourceType.getElementType());
    if (!sourceElement || !sourceElement.isSignless() ||
        (sourceElement.getWidth() != 8 && sourceElement.getWidth() != 16 && sourceElement.getWidth() != 32))
        return op->emitOpError("requires a signed i8/i16/i32 source vector");
    if (sourceType != resultType)
        return op->emitOpError("requires source and result to have identical VectorSSA types");
    return verifyMaskMatchesVector(op, mask, sourceType);
}

template <typename OpTy>
static mlir::LogicalResult verifyVectorShiftOp(OpTy op)
{
    if (!hasEnclosingRegion<VecFuncOp>(op.getOperation()))
        return op.emitOpError("must be nested inside a tla.vec.func region");
    if (mlir::failed(verifyShiftSourceAndResult(
            op.getOperation(), op.getLhs().getType(), op.getResult().getType(), op.getMask())))
        return mlir::failure();

    VectorSSAType sourceType = op.getLhs().getType();
    VectorSSAType shiftType = op.getRhs().getType();
    auto sourceElement = mlir::cast<mlir::IntegerType>(sourceType.getElementType());
    auto shiftElement = mlir::dyn_cast<mlir::IntegerType>(shiftType.getElementType());
    if (!shiftElement || !shiftElement.isSignless() || shiftElement.getWidth() != sourceElement.getWidth())
        return op.emitOpError() << "requires a signed shift vector with the source element width ("
                                << sourceElement.getWidth() << " bits)";
    if (shiftType.getValidLanes() != sourceType.getValidLanes())
        return op.emitOpError("requires source and shift vectors to have the same valid lane count");
    return mlir::success();
}

template <typename OpTy>
static mlir::LogicalResult verifyScalarShiftOp(OpTy op)
{
    if (!hasEnclosingRegion<VecFuncOp>(op.getOperation()))
        return op.emitOpError("must be nested inside a tla.vec.func region");
    if (mlir::failed(verifyShiftSourceAndResult(
            op.getOperation(), op.getLhs().getType(), op.getResult().getType(), op.getMask())))
        return mlir::failure();
    if (!op.getRhs().getType().isSignlessInteger(16))
        return op.emitOpError("requires an i16 scalar shift amount");
    return mlir::success();
}

mlir::LogicalResult ShiftLeftOp::verify()
{
    return verifyVectorShiftOp(*this);
}

mlir::LogicalResult ShiftRightOp::verify()
{
    return verifyVectorShiftOp(*this);
}

mlir::LogicalResult ShiftLeftsOp::verify()
{
    return verifyScalarShiftOp(*this);
}

mlir::LogicalResult ShiftRightsOp::verify()
{
    return verifyScalarShiftOp(*this);
}

#define TLA_VERIFY_MASKED_VECTOR_OPERAND(OpTy)                                             \
    mlir::LogicalResult OpTy::verify()                                                     \
    {                                                                                      \
        if (!hasEnclosingRegion<VecFuncOp>(getOperation()))                                \
            return emitOpError("must be nested inside a tla.vec.func region");             \
        return verifyMaskMatchesVector(getOperation(), getMask(), getOperand().getType()); \
    }

TLA_VERIFY_MASKED_VECTOR_OPERAND(ExpOp)
TLA_VERIFY_MASKED_VECTOR_OPERAND(LogOp)
TLA_VERIFY_MASKED_VECTOR_OPERAND(SqrtOp)
TLA_VERIFY_MASKED_VECTOR_OPERAND(AbsOp)
TLA_VERIFY_MASKED_VECTOR_OPERAND(NegOp)

#undef TLA_VERIFY_MASKED_VECTOR_OPERAND

mlir::LogicalResult StoreOp::verify()
{
    if (!hasEnclosingRegion<VecFuncOp>(getOperation()))
        return emitOpError("must be nested inside a tla.vec.func region");
    auto destType = mlir::dyn_cast<TlaTensorType>(getDest().getType());
    if (!destType)
        return emitOpError("dest must be !tla.tensor");
    if (destType.getPtr().getAddrspace() != AddressSpace::ub)
        return emitOpError("dest !tla.tensor must be in ub address space");

    if (mlir::isa<MaskSSAType>(getSource().getType())) {
        if (getMask())
            return emitOpError("predicate mask is not supported when storing !tla.mask");
        if (getUnalignedUbAccess())
            return emitOpError("unaligned_ub_access is not supported when storing !tla.mask");
        int64_t destElemBytes = getByteSizeOfFixedWidthScalarType(destType.getPtr().getPointee());
        if (destElemBytes != 1 && destElemBytes != 2 && destElemBytes != 4)
            return emitOpError(
                "dest !tla.tensor element type must be a 1/2/4-byte scalar "
                "for MaskSSA store");
        return mlir::success();
    }

    auto vectorSource = mlir::dyn_cast<VectorSSAType>(getSource().getType());
    if (!vectorSource)
        return emitOpError("source must be !tla.vector or !tla.mask");

    bool isUnaligned = getUnalignedUbAccess().value_or(false);
    if (isUnaligned && getBlockStride())
        return emitOpError("unaligned_ub_access and block_stride are mutually exclusive");

    if (auto storeDist = getStoreDist()) {
        ::StoreDist dist = storeDist->getStoreDist();
        bool isNormal = dist == ::StoreDist::norm;
        bool isFirstElement = dist == ::StoreDist::first_element_b8 || dist == ::StoreDist::first_element_b16 ||
                              dist == ::StoreDist::first_element_b32;
        if (!isNormal && isUnaligned)
            return emitOpError("non-normal store_dist is incompatible with unaligned_ub_access");
        if (!isNormal && getBlockStride())
            return emitOpError("non-normal store_dist is incompatible with block_stride");
        if (isFirstElement && getMask())
            return emitOpError("predicate mask is not supported with first_element store_dist");

        auto destElementType = destType.getPtr().getPointee();
        // Explicit masks retain their historical behavior: their predicate
        // geometry is defined by the user and verified below against the
        // source vector. Omitted masks and mask-ignoring first-element stores
        // need compiler-derived geometry and therefore require a supported
        // source/destination width pairing.
        if ((!getMask() || isFirstElement || dist == ::StoreDist::pack4_b32) &&
            !getPackedStorePredicateGeometry(dist, vectorSource.getElementType(), destElementType)) {
            return emitOpError() << "unsupported " << stringifyStoreDist(dist)
                                 << " source/destination element widths: source "
                                 << getBitSizeOfFixedWidthScalarType(vectorSource.getElementType())
                                 << " bits, destination " << getBitSizeOfFixedWidthScalarType(destElementType)
                                 << " bits; supported layouts are pack_b32 32->16, pack_b32 16->16, pack_b16 16->8, "
                                    "pack4_b32 {8,16,32}->8, first_element_b8 8->8, first_element_b16 16->16, "
                                    "and first_element_b32 32->32";
        }
    }
    return verifyMaskMatchesVector(getOperation(), getMask(), vectorSource);
}

static mlir::LogicalResult verifyMaskProducerType(mlir::Operation* op, MaskSSAType maskType, mlir::Type elementType)
{
    int64_t elementBytes = getByteSizeOfFixedWidthScalarType(elementType);
    constexpr int64_t kVectorRegisterBytes = 256;
    if (elementBytes <= 0 || kVectorRegisterBytes % elementBytes != 0)
        return op->emitOpError() << "unsupported predicate dtype " << elementType;
    int64_t expectedPhysicalLanes = kVectorRegisterBytes / elementBytes;
    if (maskType.getPhysicalLanes() != expectedPhysicalLanes)
        return op->emitOpError() << "result mask has " << maskType.getPhysicalLanes() << " predicate lanes, expected "
                                 << expectedPhysicalLanes << " for dtype " << elementType;
    return mlir::success();
}

mlir::LogicalResult CreateMaskOp::verify()
{
    if (!hasEnclosingRegion<VecFuncOp>(getOperation()))
        return emitOpError("must be nested inside a tla.vec.func region");
    return verifyMaskProducerType(getOperation(), getResult().getType(), getDtype());
}

mlir::LogicalResult UpdateMaskOp::verify()
{
    if (!hasEnclosingRegion<VecFuncOp>(getOperation()))
        return emitOpError("must be nested inside a tla.vec.func region");
    return verifyMaskProducerType(getOperation(), getMask().getType(), getDtype());
}

static mlir::LogicalResult verifyMaskSelectTypes(
    mlir::Operation* op, mlir::Type mask, mlir::Type x, mlir::Type y, mlir::Type result)
{
    auto maskType = mlir::dyn_cast<MaskSSAType>(mask);
    auto xType = mlir::dyn_cast<MaskSSAType>(x);
    auto yType = mlir::dyn_cast<MaskSSAType>(y);
    auto resultType = mlir::dyn_cast<MaskSSAType>(result);
    if (!maskType || !xType || !yType || !resultType)
        return op->emitOpError("requires mask, x, y, and result to all be MaskSSA");
    if (maskType != xType || yType != xType || resultType != xType)
        return op->emitOpError("requires mask, x, y, and result to have the same MaskSSA type");
    return mlir::success();
}

mlir::LogicalResult WhereOp::verify()
{
    if (!hasEnclosingRegion<VecFuncOp>(getOperation()))
        return emitOpError("must be nested inside a tla.vec.func region");

    bool xIsVector = mlir::isa<VectorSSAType>(getX().getType());
    bool yIsVector = mlir::isa<VectorSSAType>(getY().getType());
    bool resultIsVector = mlir::isa<VectorSSAType>(getResult().getType());
    if (xIsVector != yIsVector || xIsVector != resultIsVector)
        return emitOpError("requires x, y, and result to all be VectorSSA or all be MaskSSA");

    if (xIsVector)
        return verifyMaskMatchesVector(getOperation(), getMask(), mlir::cast<VectorSSAType>(getX().getType()));

    return verifyMaskSelectTypes(
        getOperation(), getMask().getType(), getX().getType(), getY().getType(), getResult().getType());
}

mlir::LogicalResult GatherOp::verify()
{
    if (!hasEnclosingRegion<VecFuncOp>(getOperation()))
        return emitOpError("must be nested inside a tla.vec.func region");

    auto indexType = mlir::dyn_cast<VectorSSAType>(getY().getType());
    if (!indexType)
        return emitOpError("gather indices must be !tla.vector");
    auto resultTy = mlir::dyn_cast<VectorSSAType>(getResult().getType());
    if (!resultTy)
        return emitOpError("gather result must be !tla.vector");

    // Reg→Reg: x is VectorSSA; lowers via bc vselr. No mask.
    if (auto srcTy = mlir::dyn_cast<VectorSSAType>(getX().getType())) {
        if (getMask())
            return emitOpError("Reg→Reg gather does not support mask");
        if (resultTy != srcTy)
            return emitOpError("result type must match src vector type");

        mlir::Type srcElem = srcTy.getElementType();
        unsigned srcBytes = 0;
        if (auto intTy = mlir::dyn_cast<mlir::IntegerType>(srcElem)) {
            if (intTy.getWidth() != 8 && intTy.getWidth() != 16 && intTy.getWidth() != 32)
                return emitOpError("src element width must be 8/16/32 bits");
            srcBytes = intTy.getWidth() / 8;
        } else if (srcElem.isF16() || srcElem.isBF16()) {
            srcBytes = 2;
        } else if (srcElem.isF32()) {
            srcBytes = 4;
        } else {
            return emitOpError("unsupported src element type for Reg→Reg gather");
        }

        auto indexElem = mlir::dyn_cast<mlir::IntegerType>(indexType.getElementType());
        if (!indexElem || indexElem.getWidth() != srcBytes * 8)
            return emitOpError(
                "Reg→Reg gather index must be an integer with the same bit-width as "
                "src element (i8/u8, i16/u16, or i32/u32)");

        if (srcTy.getValidLanes() != mlir::ShapedType::kDynamic &&
            indexType.getValidLanes() != mlir::ShapedType::kDynamic &&
            srcTy.getValidLanes() != indexType.getValidLanes())
            return emitOpError("src and index must have the same lane count");

        return mlir::success();
    }

    // UB→Reg: x is a tensor; indices must be i32; optional mask.
    if (!mlir::isa<TlaTensorType>(getX().getType()))
        return emitOpError("gather source must be a UB tensor or !tla.vector");

    auto indexElementType = indexType.getElementType();
    auto integerType = mlir::dyn_cast<mlir::IntegerType>(indexElementType);
    if (!integerType || !integerType.isSignless() || integerType.getWidth() != 32)
        return emitOpError() << "gather indices must be i32, got " << indexElementType;
    return verifyMaskMatchesVector(getOperation(), getMask(), getResult().getType());
}

mlir::LogicalResult SqueezeOp::verify()
{
    if (!hasEnclosingRegion<VecFuncOp>(getOperation()))
        return emitOpError("must be nested inside a tla.vec.func region");
    return verifyMaskMatchesVector(getOperation(), getMask(), getSrc().getType());
}

mlir::LogicalResult HistogramsOp::verify()
{
    if (!hasEnclosingRegion<VecFuncOp>(getOperation()))
        return emitOpError("must be nested inside a tla.vec.func region");

    auto srcTy = mlir::dyn_cast<VectorSSAType>(getSrc().getType());
    if (!srcTy)
        return emitOpError("src must be !tla.vector");
    auto srcElem = mlir::dyn_cast<mlir::IntegerType>(srcTy.getElementType());
    if (!srcElem || srcElem.getWidth() != 8)
        return emitOpError("src element type must be i8 or ui8");

    auto dstTy = mlir::dyn_cast<VectorSSAType>(getDst().getType());
    if (!dstTy)
        return emitOpError("dst must be !tla.vector");
    auto dstElem = mlir::dyn_cast<mlir::IntegerType>(dstTy.getElementType());
    if (!dstElem || dstElem.getWidth() != 16)
        return emitOpError("dst element type must be i16 or ui16");

    auto resultTy = mlir::dyn_cast<VectorSSAType>(getResult().getType());
    if (!resultTy)
        return emitOpError("result must be !tla.vector");
    if (resultTy != dstTy)
        return emitOpError("dst type must match result type");

    // VL = 256B: 256 x i8 in, 128 x i16 out.
    if (srcTy.getValidLanes() != mlir::ShapedType::kDynamic && srcTy.getValidLanes() != 256)
        return emitOpError("src must have 256 lanes (full VL for i8)");
    if (dstTy.getValidLanes() != mlir::ShapedType::kDynamic && dstTy.getValidLanes() != 128)
        return emitOpError("dst must have 128 lanes (full VL for i16)");

    if (getMask())
        return verifyMaskMatchesVector(getOperation(), getMask(), getSrc().getType());
    return mlir::success();
}

mlir::LogicalResult CastOp::verify()
{
    if (!hasEnclosingRegion<VecFuncOp>(getOperation()))
        return emitOpError("must be nested inside a tla.vec.func region");
    auto source = getSource().getType().getElementType();
    auto dest = getResult().getType().getElementType();
    auto trait = getTrait();
    if (trait.size() == 3 && trait[2] == static_cast<int32_t>(::RoundMode::cast_hybrid) &&
        !(source.isF32() && mlir::isa<HiFloat8Type>(dest)))
        return emitOpError("CAST_HYBRID only supports Float32->HiFloat8");
    if (isa<::tla::HiFloat8Type>(source) || isa<::tla::HiFloat8Type>(dest)) {
        if (trait.size() != 3 || trait[0] < 0 || trait[0] > 3 || trait[1] < 0 || trait[1] > 2 || trait[2] < 0 ||
            !::symbolizeRoundMode(trait[2]))
            return emitOpError("trait must contain valid register-slot, saturation and rounding codes");
        if (!((source.isF32() && mlir::isa<HiFloat8Type>(dest)) || (mlir::isa<HiFloat8Type>(source) && dest.isF32())))
            return emitOpError("HiFloat8 pairs with Float32; compose other routes explicitly");
        if (mlir::isa<HiFloat8Type>(dest) && trait[2] != static_cast<int32_t>(::RoundMode::cast_round) &&
            trait[2] != static_cast<int32_t>(::RoundMode::cast_hybrid))
            return emitOpError("FP32->HiF8 supports CAST_ROUND or CAST_HYBRID");
    }
    return verifyMaskMatchesVector(getOperation(), getMask(), getSource().getType());
}

mlir::LogicalResult BitCastOp::verify()
{
    if (!hasEnclosingRegion<VecFuncOp>(getOperation()))
        return emitOpError("must be nested inside a tla.vec.func region");

    VectorSSAType sourceType = getSource().getType();
    VectorSSAType resultType = getResult().getType();
    if (sourceType.getValidLanes() != resultType.getValidLanes())
        return emitOpError() << "preserves the lane count, got " << sourceType.getValidLanes() << " source lane(s) and "
                             << resultType.getValidLanes() << " result lane(s)";

    int64_t sourceBits = getBitSizeOfFixedWidthScalarType(sourceType.getElementType());
    int64_t resultBits = getBitSizeOfFixedWidthScalarType(resultType.getElementType());
    if (sourceBits <= 0 || resultBits <= 0)
        return emitOpError() << "unsupported element type: source " << sourceType.getElementType() << ", result "
                             << resultType.getElementType();
    if (sourceBits != resultBits)
        return emitOpError() << "requires source and result element types of the same width, got "
                             << sourceType.getElementType() << " (" << sourceBits << " bits) and "
                             << resultType.getElementType() << " (" << resultBits << " bits)";

    return mlir::success();
}

mlir::LogicalResult ReduceOp::verify()
{
    if (!hasEnclosingRegion<VecFuncOp>(getOperation()))
        return emitOpError("must be nested inside a tla.vec.func region");
    return verifyMaskMatchesVector(getOperation(), getMask(), getOperand().getType());
}

mlir::LogicalResult ReduceIntraBlockOp::verify()
{
    if (!hasEnclosingRegion<VecFuncOp>(getOperation()))
        return emitOpError("must be nested inside a tla.vec.func region");

    if ((getKind() != "add") && (getKind() != "max") && (getKind() != "min"))
        return emitOpError() << "kind must be add, max, or min, got \"" << getKind() << "\"";

    auto operandTy = mlir::dyn_cast<VectorSSAType>(getOperand().getType());
    if (!operandTy)
        return emitOpError("operand must be !tla.vector");
    auto resultTy = mlir::dyn_cast<VectorSSAType>(getResult().getType());
    if (!resultTy)
        return emitOpError("result must be !tla.vector");
    if (resultTy != operandTy) {
        // ADD on 16-bit sources widens to a 32-bit result (the same widening
        // behavior as vcadd): the Vcgadd dst is 32-bit, and the DSL returns it
        // as-is instead of casting back, so block sums may exceed the source
        // type's representable range and stay exact in i32/ui32.
        auto operandElem = mlir::dyn_cast<mlir::IntegerType>(operandTy.getElementType());
        bool widenAdd = (getKind() == "add") && operandElem && (operandElem.getWidth() == 16);
        if (!widenAdd)
            return emitOpError("result type must match operand type (packed mirror)");
        // The widening keeps the operand's sign category: i16 -> i32, ui16 -> ui32.
        auto resultElem = mlir::dyn_cast<mlir::IntegerType>(resultTy.getElementType());
        if (!resultElem || (resultElem.getWidth() != 32) || (resultElem.getSignedness() != operandElem.getSignedness()))
            return emitOpError() << "add on i16/ui16 widens the result to the matching 32-bit type "
                                 << "(i16->i32, ui16->ui32), got " << resultTy.getElementType();
        // Lane count halves per 32-bit result lane, rounding up so a tail's
        // last valid element keeps its widened lane (one valid i16 lane still
        // yields one valid i32 result lane).
        if ((operandTy.getValidLanes() != mlir::ShapedType::kDynamic) &&
            (resultTy.getValidLanes() != ((operandTy.getValidLanes() + 1) / 2)))
            return emitOpError() << "add on i16/ui16 keeps the 256B container, expected "
                                 << ((operandTy.getValidLanes() + 1) / 2) << " result lane(s), got "
                                 << resultTy.getValidLanes();
    }

    // The Vcg group-reduction instructions cover f16/f32 and 16/32-bit integers
    // (S/U variants). i8/u8/i64/bf16/f8 are not supported.
    mlir::Type elem = operandTy.getElementType();
    bool supported = mlir::isa<mlir::Float16Type, mlir::Float32Type>(elem);
    if (auto intType = mlir::dyn_cast<mlir::IntegerType>(elem))
        supported = (intType.getWidth() == 16) || (intType.getWidth() == 32);
    if (!supported)
        return emitOpError() << "unsupported reduce_intra_block element type " << elem
                             << " (supported: f16/f32/i16/i32/ui16/ui32)";

    return verifyMaskMatchesVector(getOperation(), getMask(), getOperand().getType());
}

mlir::LogicalResult LoadOp::verify()
{
    if (!hasEnclosingRegion<VecFuncOp>(getOperation()))
        return emitOpError("must be nested inside a tla.vec.func region");

    auto sourceType = mlir::dyn_cast<TlaTensorType>(getSource().getType());
    if (!sourceType)
        return emitOpError("source must be !tla.tensor");
    if (sourceType.getPtr().getAddrspace() != AddressSpace::ub)
        return emitOpError("source !tla.tensor must be in ub address space");

    if (auto maskResult = mlir::dyn_cast<MaskSSAType>(getResult().getType())) {
        if (getResult2())
            return emitOpError("second result is not valid when loading !tla.mask");
        if (getLoadDist()) {
            auto dist = getLoadDist()->getLoadDist();
            if (dist != LoadDist::norm && dist != LoadDist::us && dist != LoadDist::ds)
                return emitOpError(
                    "mask load only supports load_dist norm|us|ds "
                    "(AscendC MaskDist NORM/US/DS)");
            if (dist == LoadDist::us || dist == LoadDist::ds) {
                // AscendC LoadAlign<T, MaskDist>: SupportBytes<T, 1, 2, 4, 8>.
                int64_t sourceElemBytes = getByteSizeOfFixedWidthScalarType(sourceType.getPtr().getPointee());
                if (sourceElemBytes != 1 && sourceElemBytes != 2 && sourceElemBytes != 4 && sourceElemBytes != 8)
                    return emitOpError(
                        "MaskLoadDist US/DS requires a b8/b16/b32/b64 scalar UB "
                        "(AscendC LoadAlign<T, MaskDist>)");
            }
        }
        if (getBlockStride() || getRepeatStride())
            return emitOpError("block_stride/repeat_stride is not supported when loading !tla.mask");
        if (getUnalignedUbAccess())
            return emitOpError("unaligned_ub_access is not supported when loading !tla.mask");
        int64_t sourceElemBytes = getByteSizeOfFixedWidthScalarType(sourceType.getPtr().getPointee());
        bool usOrDs = false;
        if (getLoadDist()) {
            auto dist = getLoadDist()->getLoadDist();
            usOrDs = dist == LoadDist::us || dist == LoadDist::ds;
        }
        if (usOrDs) {
            if (sourceElemBytes != 1 && sourceElemBytes != 2 && sourceElemBytes != 4 && sourceElemBytes != 8)
                return emitOpError(
                    "source !tla.tensor element type must be a b8/b16/b32/b64 "
                    "scalar for MaskLoadDist US/DS");
        } else if (sourceElemBytes != 1 && sourceElemBytes != 2 && sourceElemBytes != 4) {
            return emitOpError(
                "source !tla.tensor element type must be a 1/2/4-byte scalar "
                "for MaskSSA load");
        }
        int64_t lanes = maskResult.getPhysicalLanes();
        if (lanes != 32 && lanes != 64 && lanes != 128 && lanes != 256)
            return emitOpError() << "unsupported !tla.mask lane count " << lanes;
        return mlir::success();
    }

    if (!mlir::isa<VectorSSAType>(getResult().getType()))
        return emitOpError("result must be !tla.vector or !tla.mask");

    if (getLoadDist() && getLoadDist()->getLoadDist() == ::LoadDist::unpack4_b8) {
        auto resultType = mlir::cast<VectorSSAType>(getResult().getType());
        if (getByteSizeOfFixedWidthScalarType(sourceType.getPtr().getPointee()) != 1 ||
            getByteSizeOfFixedWidthScalarType(resultType.getElementType()) != 1)
            return emitOpError("unpack4_b8 requires 1-byte source and result element types");
        if (getUnalignedUbAccess().value_or(false))
            return emitOpError("unpack4_b8 requires aligned UB access");
        if (resultType.getValidLanes() != 256)
            return emitOpError("unpack4_b8 produces a full 256-byte register");
    }

    if (auto blockStrideAttr = getBlockStrideAttr()) {
        if (getLoadDist())
            return emitOpError("block_stride is mutually exclusive with load_dist");
        if (getUnalignedUbAccess())
            return emitOpError("block_stride is mutually exclusive with unaligned_ub_access");
        if (getResult2())
            return emitOpError("block_stride is not valid with a second (dual-destination) result");
        if (blockStrideAttr.getInt() < 0 || blockStrideAttr.getInt() > 0xFFFF)
            return emitOpError("block_stride must fit the 16-bit vsldb immediate field [0, 65535]");
    }
    if (auto repeatStrideAttr = getRepeatStrideAttr()) {
        if (!getBlockStrideAttr())
            return emitOpError("repeat_stride requires block_stride (POST_MODE_NORMAL pre-offset)");
        if (repeatStrideAttr.getInt() < 0 || repeatStrideAttr.getInt() > 0xFFFF)
            return emitOpError("repeat_stride must fit the 16-bit vsldb immediate field [0, 65535]");
    }

    bool isDintlv = false;
    if (auto loadDistAttr = getLoadDist())
        isDintlv = loadDistAttr->getLoadDist() == ::LoadDist::dintlv_b32;

    mlir::Value result2 = getResult2();
    bool result2IsVector = result2 && mlir::isa<VectorSSAType>(result2.getType());
    if (isDintlv) {
        if (getUnalignedUbAccess().value_or(false))
            return emitOpError("load_dist dintlv_b32 requires aligned UB access");
        if (!result2IsVector)
            return emitOpError("load_dist dintlv_b32 requires a second !tla.vector result");
    } else if (result2) {
        return emitOpError("second result is only valid as !tla.vector (dintlv_b32)");
    }
    if (getBlockStride() && getUnalignedUbAccess())
        return emitOpError("block_stride is not supported with unaligned_ub_access");
    return mlir::success();
}

static mlir::LogicalResult verifyUpdateStride(mlir::Operation* op, mlir::Value stride)
{
    if (!stride)
        return op->emitOpError("requires post_update_stride");
    if (auto constantOp = stride.getDefiningOp<mlir::arith::ConstantOp>()) {
        if (auto intAttr = mlir::dyn_cast<mlir::IntegerAttr>(constantOp.getValue())) {
            if (intAttr.getInt() <= 0)
                return op->emitOpError("post_update_stride must be > 0");
        }
    }
    return mlir::success();
}

static mlir::LogicalResult verifyUpdatePtr(mlir::Operation* op, mlir::Value ptr, mlir::Type elemType)
{
    auto ptrTy = mlir::dyn_cast<PtrType>(ptr.getType());
    if (!ptrTy)
        return op->emitOpError("next-window result must be !tla.ptr");
    if (ptrTy.getAddrspace() != AddressSpace::ub)
        return op->emitOpError("next-window !tla.ptr must be ub");
    if (ptrTy.getPointee() != elemType)
        return op->emitOpError("next-window !tla.ptr pointee must match the tensor element type");
    return mlir::success();
}

static mlir::LogicalResult verifyOptionalBlockStride(mlir::Operation* op, mlir::Value stride)
{
    if (!stride)
        return mlir::success();
    if (auto constantOp = stride.getDefiningOp<mlir::arith::ConstantOp>()) {
        if (auto intAttr = mlir::dyn_cast<mlir::IntegerAttr>(constantOp.getValue())) {
            int64_t value = intAttr.getInt();
            if (value < 0 || value > 0xFFFF)
                return op->emitOpError("block_stride must be in [0, 65535]");
        }
    }
    return mlir::success();
}

static mlir::LogicalResult verifyDataBlockUpdateStride(mlir::Operation* op, mlir::Value stride)
{
    if (failed(verifyUpdateStride(op, stride)))
        return mlir::failure();
    if (auto constantOp = stride.getDefiningOp<mlir::arith::ConstantOp>()) {
        if (auto intAttr = mlir::dyn_cast<mlir::IntegerAttr>(constantOp.getValue())) {
            if (intAttr.getInt() > 0xFFFF)
                return op->emitOpError("DATA_BLOCK_COPY post_update_stride must be in [1, 65535]");
        }
    }
    return mlir::success();
}

mlir::LogicalResult LoadUpdateOp::verify()
{
    if (!hasEnclosingRegion<VecFuncOp>(getOperation()))
        return emitOpError("must be nested inside a tla.vec.func region");
    auto sourceType = mlir::dyn_cast<TlaTensorType>(getSource().getType());
    if (!sourceType)
        return emitOpError("source must be !tla.tensor");
    if (sourceType.getPtr().getAddrspace() != AddressSpace::ub)
        return emitOpError("source !tla.tensor must be in ub address space");
    if (!mlir::isa<VectorSSAType>(getResult().getType()))
        return emitOpError("result must be !tla.vector");
    if (failed(verifyOptionalBlockStride(getOperation(), getBlockStride())))
        return mlir::failure();
    if (getBlockStride()) {
        if (failed(verifyDataBlockUpdateStride(getOperation(), getPostUpdateStride())))
            return mlir::failure();
    } else if (failed(verifyUpdateStride(getOperation(), getPostUpdateStride()))) {
        return mlir::failure();
    }
    return verifyUpdatePtr(getOperation(), getNextPtr(), sourceType.getPtr().getPointee());
}

mlir::LogicalResult StoreUpdateOp::verify()
{
    if (!hasEnclosingRegion<VecFuncOp>(getOperation()))
        return emitOpError("must be nested inside a tla.vec.func region");
    auto destType = mlir::dyn_cast<TlaTensorType>(getDest().getType());
    if (!destType)
        return emitOpError("dest must be !tla.tensor");
    if (destType.getPtr().getAddrspace() != AddressSpace::ub)
        return emitOpError("dest !tla.tensor must be in ub address space");
    if (!mlir::isa<VectorSSAType>(getSource().getType()))
        return emitOpError("source must be !tla.vector");
    if (failed(verifyOptionalBlockStride(getOperation(), getBlockStride())))
        return mlir::failure();
    if (getBlockStride()) {
        if (failed(verifyDataBlockUpdateStride(getOperation(), getPostUpdateStride())))
            return mlir::failure();
    } else if (failed(verifyUpdateStride(getOperation(), getPostUpdateStride()))) {
        return mlir::failure();
    }
    if (failed(verifyUpdatePtr(getOperation(), getNextPtr(), destType.getPtr().getPointee())))
        return mlir::failure();
    return verifyMaskMatchesVector(getOperation(), getMask(), getSource().getType());
}

static mlir::LogicalResult verifyUnalignUbTensor(mlir::Operation* op, mlir::Value tensor, llvm::StringRef name)
{
    if (!hasEnclosingRegion<VecFuncOp>(op))
        return op->emitOpError("must be nested inside a tla.vec.func region");
    auto tensorType = mlir::dyn_cast<TlaTensorType>(tensor.getType());
    if (!tensorType)
        return op->emitOpError() << name << " must be !tla.tensor";
    if (tensorType.getPtr().getAddrspace() != AddressSpace::ub)
        return op->emitOpError() << name << " !tla.tensor must be in ub address space";
    return mlir::success();
}

mlir::LogicalResult LoadUnalignUpdateOp::verify()
{
    if (failed(verifyUnalignUbTensor(getOperation(), getSource(), "source")))
        return mlir::failure();
    if (!mlir::isa<VectorSSAType>(getResult().getType()))
        return emitOpError("result must be !tla.vector");
    if (failed(verifyUpdateStride(getOperation(), getPostUpdateStride())))
        return mlir::failure();
    auto sourceType = mlir::cast<TlaTensorType>(getSource().getType());
    return verifyUpdatePtr(getOperation(), getNextPtr(), sourceType.getPtr().getPointee());
}

mlir::LogicalResult StoreUnalignUpdateOp::verify()
{
    if (failed(verifyUnalignUbTensor(getOperation(), getDest(), "dest")))
        return mlir::failure();
    if (!mlir::isa<VectorSSAType>(getSource().getType()))
        return emitOpError("source must be !tla.vector");
    if (failed(verifyUpdateStride(getOperation(), getPostUpdateStride())))
        return mlir::failure();
    auto destType = mlir::cast<TlaTensorType>(getDest().getType());
    return verifyUpdatePtr(getOperation(), getNextPtr(), destType.getPtr().getPointee());
}

static mlir::LogicalResult verifyConstantScalarIndexInBounds(mlir::Operation* op, mlir::Value index, int64_t length)
{
    if (length == mlir::ShapedType::kDynamic)
        return mlir::success();
    auto constantOp = index.getDefiningOp<mlir::arith::ConstantOp>();
    if (!constantOp)
        return mlir::success();
    auto indexAttr = mlir::dyn_cast<mlir::IntegerAttr>(constantOp.getValue());
    if (!indexAttr)
        return mlir::success();
    int64_t constantIndex = indexAttr.getInt();
    if (constantIndex < 0 || constantIndex >= length)
        return op->emitOpError() << "index " << constantIndex << " is out of bounds for length " << length;
    return mlir::success();
}

mlir::LogicalResult BitwiseNotOp::verify()
{
    if (!hasEnclosingRegion<VecFuncOp>(getOperation()))
        return emitOpError("must be nested inside a tla.vec.func region");

    auto operandVector = mlir::dyn_cast<VectorSSAType>(getOperand().getType());
    auto resultVector = mlir::dyn_cast<VectorSSAType>(getResult().getType());
    bool operandIsVector = static_cast<bool>(operandVector);
    bool resultIsVector = static_cast<bool>(resultVector);
    if (operandIsVector != resultIsVector)
        return emitOpError("requires operand and result to have the same !tla.vector or !tla.mask category");

    if (operandIsVector) {
        if (operandVector.getElementType() != resultVector.getElementType())
            return emitOpError("requires !tla.vector operand and result to have identical element types");
        return verifyMaskMatchesVector(getOperation(), getMask(), operandVector);
    }

    auto operandMask = mlir::cast<MaskSSAType>(getOperand().getType());
    auto resultMask = mlir::cast<MaskSSAType>(getResult().getType());
    if (resultMask != operandMask)
        return emitOpError("requires MaskSSA operand and result to have identical types");
    if (getMask() && getMask().getType() != operandMask)
        return emitOpError("requires optional mask to have the same MaskSSA type as operand");
    return mlir::success();
}

template <typename OpTy>
static mlir::LogicalResult verifyBitwiseBinaryOp(OpTy op)
{
    if (!hasEnclosingRegion<VecFuncOp>(op.getOperation()))
        return op.emitOpError("must be nested inside a tla.vec.func region");

    auto lhsVector = mlir::dyn_cast<VectorSSAType>(op.getLhs().getType());
    auto rhsVector = mlir::dyn_cast<VectorSSAType>(op.getRhs().getType());
    auto resultVector = mlir::dyn_cast<VectorSSAType>(op.getResult().getType());
    bool lhsIsVector = static_cast<bool>(lhsVector);
    bool rhsIsVector = static_cast<bool>(rhsVector);
    bool resultIsVector = static_cast<bool>(resultVector);
    if (lhsIsVector != rhsIsVector || lhsIsVector != resultIsVector)
        return op.emitOpError("requires lhs, rhs, and result to have the same !tla.vector or !tla.mask category");

    if (lhsIsVector) {
        mlir::Type lhsElementType = lhsVector.getElementType();
        if (rhsVector.getElementType() != lhsElementType || resultVector.getElementType() != lhsElementType)
            return op.emitOpError("requires !tla.vector lhs, rhs, and result to have identical element types");
        return verifyMaskMatchesVector(op.getOperation(), op.getMask(), lhsVector);
    }

    auto lhsMask = mlir::cast<MaskSSAType>(op.getLhs().getType());
    auto rhsMask = mlir::cast<MaskSSAType>(op.getRhs().getType());
    auto resultMask = mlir::cast<MaskSSAType>(op.getResult().getType());
    if (rhsMask != lhsMask || resultMask != lhsMask)
        return op.emitOpError("requires MaskSSA lhs, rhs, and result to have identical types");
    if (op.getMask() && op.getMask().getType() != lhsMask)
        return op.emitOpError("requires optional mask to have the same MaskSSA type as operands");
    return mlir::success();
}

mlir::LogicalResult BitwiseAndOp::verify()
{
    return verifyBitwiseBinaryOp(*this);
}

mlir::LogicalResult BitwiseOrOp::verify()
{
    return verifyBitwiseBinaryOp(*this);
}

mlir::LogicalResult BitwiseXorOp::verify()
{
    return verifyBitwiseBinaryOp(*this);
}

mlir::LogicalResult ArangeOp::verify()
{
    if (!hasEnclosingRegion<VecFuncOp>(getOperation()))
        return emitOpError("must be nested inside a tla.vec.func region");
    auto order = getOrderAttr().getValue();
    if (order != "increase" && order != "decrease")
        return emitOpError("unsupported arange order '") << order << "'; expected 'increase' or 'decrease'";
    return mlir::success();
}

mlir::LogicalResult FlagOp::verify()
{
    int64_t id = getId();
    if (id < -1 || id > 7)
        return emitOpError("id must be -1 or in range 0..7");
    return mlir::success();
}

// Synchronization/mutex/barrier ops must live inside a tla.cube or tla.vector
// region (either core-kind region; not the func-level scope).
#define TLA_VERIFY_IN_CUBE_OR_VECTOR(OpTy)                                                                \
    mlir::LogicalResult OpTy::verify()                                                                    \
    {                                                                                                     \
        if (!hasEnclosingRegion<CubeOp>(getOperation()) && !hasEnclosingRegion<VectorOp>(getOperation())) \
            return emitOpError("must be nested inside a tla.cube or tla.vector region");                  \
        return mlir::success();                                                                           \
    }

TLA_VERIFY_IN_CUBE_OR_VECTOR(SetFlagOp)
TLA_VERIFY_IN_CUBE_OR_VECTOR(WaitFlagOp)
TLA_VERIFY_IN_CUBE_OR_VECTOR(MutexLockOp)
TLA_VERIFY_IN_CUBE_OR_VECTOR(MutexUnlockOp)
TLA_VERIFY_IN_CUBE_OR_VECTOR(PipeBarrierOp)

#undef TLA_VERIFY_IN_CUBE_OR_VECTOR

// Look up ``tla.core_ratio`` on the enclosing entry func (tla.func / func.func).
// Returns the AIV half of the ratio when present; otherwise nullopt (legacy
// default allows aiv_id in {0,1}).
static std::optional<int64_t> getEnclosingMixAivRatio(mlir::Operation* op)
{
    for (mlir::Operation* parent = op->getParentOp(); parent; parent = parent->getParentOp()) {
        auto attr = parent->getAttrOfType<mlir::DenseI64ArrayAttr>("tla.core_ratio");
        if (attr && attr.size() == 2)
            return attr[1];
        if (mlir::isa<mlir::func::FuncOp, FuncOp>(parent))
            break;
    }
    return std::nullopt;
}

template <typename OpTy>
static mlir::LogicalResult verifyCrossCoreFlagOp(OpTy op)
{
    if (!hasEnclosingRegion<CubeOp>(op.getOperation()) && !hasEnclosingRegion<VectorOp>(op.getOperation()))
        return op.emitOpError("must be nested inside a tla.cube or tla.vector region");
    auto flagType = mlir::cast<CrossFlagType>(op.getFlag().getType());
    auto aivIdAttr = op->template getAttrOfType<mlir::IntegerAttr>("aiv_id");
    if (flagType.getMode() == 4) {
        std::optional<int64_t> aivRatio = getEnclosingMixAivRatio(op.getOperation());
        if (aivRatio && *aivRatio == 1) {
            if (!aivIdAttr || aivIdAttr.getInt() != 0)
                return op.emitOpError("core_ratio [1, 1] requires aiv_id to be the compile-time integer 0");
        } else if (!aivIdAttr || (aivIdAttr.getInt() != 0 && aivIdAttr.getInt() != 1)) {
            return op.emitOpError("mode 4 requires aiv_id to be the compile-time integer 0 or 1");
        }
    } else if (aivIdAttr) {
        return op.emitOpError("aiv_id is only valid for mode 4 cross flags");
    }
    return mlir::success();
}

mlir::LogicalResult CrossCoreSetFlagOp::verify()
{
    return verifyCrossCoreFlagOp(*this);
}

mlir::LogicalResult CrossCoreWaitFlagOp::verify()
{
    return verifyCrossCoreFlagOp(*this);
}

mlir::LogicalResult LocalMemBarOp::verify()
{
    auto kind = getBarrierKind();
    if (kind < 0 || kind > 11)
        return emitOpError("barrier_kind ") << kind << " is out of range [0, 11]";
    if (!hasEnclosingRegion<CubeOp>(getOperation()) && !hasEnclosingRegion<VectorOp>(getOperation()))
        return emitOpError("must be nested inside a tla.cube or tla.vector region");
    return mlir::success();
}

mlir::LogicalResult CopyOp::verify()
{
    auto srcTy = mlir::dyn_cast<TlaTensorType>(getSrc().getType());
    auto dstTy = mlir::dyn_cast<TlaTensorType>(getDst().getType());
    if (!srcTy || !dstTy)
        return mlir::success(); // Operand type verifier handles malformed tensors.
    AddressSpace src = srcTy.getPtr().getAddrspace();
    AddressSpace dst = dstTy.getPtr().getAddrspace();

    // Cube data-path copies: GM->L1, L1->L0A, L1->L0B, L0C->GM, L0C->UB, L0C->L1, L1->UB.
    bool cubeRoute =
        (src == AddressSpace::gm && dst == AddressSpace::l1) || (src == AddressSpace::l1 && dst == AddressSpace::l0a) ||
        (src == AddressSpace::l1 && dst == AddressSpace::l0b) ||
        (src == AddressSpace::l0c && dst == AddressSpace::gm) ||
        (src == AddressSpace::l0c && dst == AddressSpace::ub) ||
        (src == AddressSpace::l0c && dst == AddressSpace::l1) || (src == AddressSpace::l1 && dst == AddressSpace::ub);
    // Vector staging copies: GM->UB, UB->GM, UB->L1.
    bool vectorRoute =
        (src == AddressSpace::gm && dst == AddressSpace::ub) || (src == AddressSpace::ub && dst == AddressSpace::gm) ||
        (src == AddressSpace::ub && dst == AddressSpace::l1) || (src == AddressSpace::ub && dst == AddressSpace::ub);

    if (cubeRoute && !hasEnclosingRegion<CubeOp>(getOperation()))
        return emitOpError(
            "copy between GM/L1/L0A/L0B/L0C/UB must be nested inside "
            "a tla.cube region");
    if (vectorRoute && !hasEnclosingRegion<VectorOp>(getOperation()))
        return emitOpError("copy between GM/UB/L1 must be nested inside a tla.vector region");

    // Validate atomic_mode attribute when present.
    auto atomicModeAttr = getAtomicModeAttr();
    if (atomicModeAttr && atomicModeAttr.getAtomicMode() != AtomicMode::none) {
        if (atomicModeAttr.getAtomicMode() != AtomicMode::add)
            return emitOpError("unsupported atomic_mode; currently only 'add' is supported");

        if (dst != AddressSpace::gm)
            return emitOpError("atomic operation requires dst to be in GM address space");

        auto elemType = dstTy.getPtr().getPointee();
        if (!elemType.isF32() && !elemType.isF16() && !elemType.isBF16() && !elemType.isInteger(32) &&
            !elemType.isInteger(16) && !elemType.isInteger(8))
            return emitOpError(
                "atomic operation requires dst element type to be one of "
                "f32, f16, bf16, i32, i16, i8");
    }

    return mlir::success();
}

mlir::LogicalResult SimtCastOp::verify()
{
    static constexpr ::llvm::StringLiteral kKinds[] = {"extsi",  "extui",  "trunci", "extf",  "truncf",
                                                       "sitofp", "uitofp", "fptosi", "fptoui"};
    for (::llvm::StringLiteral kind : kKinds)
        if (getKind() == kind)
            return mlir::success();
    return emitOpError() << "unsupported cast kind \"" << getKind() << "\"";
}

mlir::LogicalResult ScalarRoundCastOp::verify()
{
    // The mode and the result type are not independent: the scalar unit has
    // four f32->i32 instructions and exactly one f32->f16 one.
    bool resultIsF16 = mlir::isa<mlir::Float16Type>(getResult().getType());
    if (resultIsF16) {
        if (getMode() == "o")
            return mlir::success();
        return emitOpError() << "an f16 result takes only mode \"o\" (round to odd), got \"" << getMode() << "\"";
    }
    static constexpr ::llvm::StringLiteral kIntModes[] = {"rn", "ra", "rd", "ru"};
    for (::llvm::StringLiteral mode : kIntModes)
        if (getMode() == mode)
            return mlir::success();
    if (getMode() == "o")
        return emitOpError() << "mode \"o\" (round to odd) produces an f16 result, not i32";
    return emitOpError() << "an i32 result takes one of rn, ra, rd, ru, got \"" << getMode() << "\"";
}

mlir::LogicalResult SimtCmpOp::verify()
{
    if (!isSupportedCmpMode(getMode()))
        return emitOpError() << "mode must be one of lt, le, gt, ge, eq, ne, got \"" << getMode() << "\"";
    return mlir::success();
}

mlir::LogicalResult CmpOp::verify()
{
    if (!hasEnclosingRegion<VecFuncOp>(getOperation()))
        return emitOpError("must be nested inside a tla.vec.func region");
    if (!isSupportedCmpMode(getMode()))
        return emitOpError() << "mode must be one of lt, le, gt, ge, eq, ne, got \"" << getMode() << "\"";
    auto lhsType = getLhs().getType();
    mlir::Type lhsElementType = lhsType.getElementType();
    if (!isSupportedCmpElementType(lhsElementType))
        return emitOpError() << "unsupported compare element type " << lhsElementType;
    if (failed(verifyMaskMatchesVector(getOperation(), getResult(), lhsType)) ||
        failed(verifyMaskMatchesVector(getOperation(), getMask(), lhsType)))
        return mlir::failure();

    auto rhsType = getRhs().getType();
    auto rhsVectorType = mlir::dyn_cast<::tla::VectorSSAType>(rhsType);
    if (!rhsVectorType) {
        if (rhsType != lhsElementType)
            return emitOpError() << "scalar operand must have element type " << lhsElementType << ", got " << rhsType;
        return mlir::success();
    }

    mlir::Type rhsElementType = rhsVectorType.getElementType();
    if (lhsElementType != rhsElementType)
        return emitOpError() << "operands must have the same element type, got " << lhsElementType << " and "
                             << rhsElementType;

    return mlir::success();
}

mlir::LogicalResult DebugPrintOp::verify()
{
    if (!hasEnclosingRegion<CubeOp>(getOperation()) && !hasEnclosingRegion<VectorOp>(getOperation()))
        return emitOpError("must be nested inside a tla.cube or tla.vector region");

    auto formatAttr = (*this)->getAttrOfType<mlir::StringAttr>("format");
    if (!formatAttr) {
        if (getValues().size() != 1)
            return emitOpError("legacy debug_print expects exactly one operand");
        auto type = getValues().front().getType();
        if (!isSupportedPrintTensorInteger(type) && !type.isF16() && !type.isF32())
            return emitOpError("expected one of ") << kPrintTensorSupportedDtypes << " scalar, got " << type;
        return mlir::success();
    }

    unsigned fieldCount = 0;
    if (failed(parseDebugPrintFormat(getOperation(), formatAttr.getValue(), fieldCount)))
        return mlir::failure();
    if (fieldCount != getValues().size())
        return emitOpError("format argument count mismatch: format has ")
               << fieldCount << " fields but got " << getValues().size() << " operands";
    for (mlir::Value value : getValues()) {
        auto type = value.getType();
        if (!isSupportedDebugPrintScalar(type))
            return emitOpError("expected one of ") << kPrintTensorSupportedDtypes << " scalar, got " << type;
    }
    return mlir::success();
}

mlir::LogicalResult PrintVectorOp::verify()
{
    if (!hasEnclosing<VecFuncOp>(getOperation()))
        return emitOpError("must be nested inside a SIMD tla.vec.func");
    auto parent = getOperation()->getParentOfType<VecFuncOp>();
    if (auto mode = parent->getAttrOfType<mlir::StringAttr>("mode");
        mode && !mode.getValue().equals_insensitive("simd"))
        return emitOpError("register printing requires SIMD mode");
    auto element = getValue().getType().getElementType();
    if (!isSupportedPrintTensorInteger(element) && !element.isF16() && !element.isF32())
        return emitOpError("unsupported register print dtype; expected ") << kPrintTensorSupportedDtypes;
    llvm::APInt length;
    if (mlir::matchPattern(getLength(), mlir::m_ConstantInt(&length))) {
        int64_t count = length.getSExtValue();
        if (count <= 0 || count > 2048 / element.getIntOrFloatBitWidth())
            return emitOpError("register prefix length is outside the physical lane capacity");
    }
    return mlir::success();
}

mlir::LogicalResult PrintTensorOp::verify()
{
    // CANN's 1 MiB debug FIFO reserves 48 bytes for the shape TLV and 72 bytes
    // for the tensor TLV. Its 32-byte payload alignment leaves 262112 f32 values:
    // floor((1 MiB - 48 - 72) / 32) * (32 / sizeof(float)).
    constexpr int64_t kMaxFloat32Elements = 262112;
    auto tensorType = getValue().getType();
    auto ptr = tensorType.getPtr();
    auto elementType = ptr.getPointee();
    if (auto vf = getOperation()->getParentOfType<VecFuncOp>()) {
        auto mode = vf->getAttrOfType<mlir::StringAttr>("mode");
        bool simt = mode && mode.getValue().equals_insensitive("simt");
        if (simt && ptr.getAddrspace() != AddressSpace::gm && ptr.getAddrspace() != AddressSpace::ub)
            return emitOpError("tensor printing inside a SIMT vec.func requires GM or UB storage");
        if (!simt && ptr.getAddrspace() != AddressSpace::ub)
            return emitOpError("tensor printing inside a SIMD vec.func requires UB storage");
    }
    if (ptr.getAddrspace() != AddressSpace::gm && ptr.getAddrspace() != AddressSpace::ub &&
        ptr.getAddrspace() != AddressSpace::l1 && ptr.getAddrspace() != AddressSpace::l0c)
        return emitOpError("requires a GM-, UB-, L1-, or L0C-resident tensor");
    if (!elementType.isF16() && !elementType.isF32() && !isSupportedPrintTensorInteger(elementType)) {
        std::string diagnostic = "unsupported dtype " + printTensorDiagnosticTypeToken(elementType) +
                                 "; supported dtypes: " + kPrintTensorSupportedDtypes.str();
        return emitOpError(diagnostic);
    }
    if (!hasEnclosingRegion<CubeOp>(getOperation()) && !hasEnclosingRegion<VectorOp>(getOperation()))
        return emitOpError("must be nested inside a tla.cube or tla.vector region");
    if (ptr.getAddrspace() == AddressSpace::ub) {
        if (!hasEnclosingRegion<VectorOp>(getOperation()))
            return emitOpError("requires UB tensors to be nested in a tla.vector region");
    }
    if (ptr.getAddrspace() == AddressSpace::l1) {
        if (!hasEnclosingRegion<CubeOp>(getOperation()))
            return emitOpError("requires L1 tensors to be nested in a tla.cube region");
        auto layoutTag = tensorType.getLayout().getLayoutTag();
        if (layoutTag != LayoutTag::zN && layoutTag != LayoutTag::nZ)
            return emitOpError("requires an L1 zN or nZ tensor");
        llvm::SmallVector<int64_t, 4> originShape;
        llvm::SmallVector<int64_t, 4> packedShape;
        llvm::SmallVector<int64_t, 4> packedStride;
        llvm::SmallVector<int64_t, 4> coords;
        if (failed(getIndexTreeLeavesForVerify(
                getOperation(), tensorType.getLayout().getOrigin(), originShape, "tensor origin shape")) ||
            failed(getIndexTreeLeavesForVerify(
                getOperation(), tensorType.getLayout().getShape(), packedShape, "tensor packed shape")) ||
            failed(getIndexTreeLeavesForVerify(
                getOperation(), tensorType.getLayout().getStride(), packedStride, "tensor packed stride")) ||
            failed(getIndexTreeLeavesForVerify(getOperation(), tensorType.getCoord(), coords, "tensor coordinates")))
            return mlir::failure();
        if (originShape.size() != 2 || packedShape.size() != 4 || packedStride.size() != 4 || coords.size() != 2)
            return emitOpError("requires a static rank-2 L1 view");
        int64_t rows = originShape[0];
        int64_t cols = originShape[1];
        int64_t row = coords[0];
        int64_t col = coords[1];
        uint64_t elementBytes = elementType.getIntOrFloatBitWidth() / 8;
        uint64_t c0Elements = 32 / elementBytes;
        if (rows < 1 || cols < 1 || row < 0 || col < 0 ||
            llvm::any_of(packedShape, [](int64_t value) { return value <= 0; }) ||
            llvm::any_of(packedStride, [](int64_t value) { return value < 0; }))
            return emitOpError("requires static valid L1 shape, stride, and coordinates");
        if (layoutTag == LayoutTag::zN) {
            if (cols != static_cast<int64_t>(c0Elements))
                return emitOpError("requires zN L1 logical C0 width ") << c0Elements << " elements";
        } else {
            if (rows != static_cast<int64_t>(c0Elements))
                return emitOpError("requires nZ L1 logical C0 width ") << c0Elements << " elements";
        }
        std::array<uint64_t, 4> packedCoord = {
            static_cast<uint64_t>(row % packedShape[0]), static_cast<uint64_t>(row / packedShape[0]),
            static_cast<uint64_t>(col % packedShape[2]), static_cast<uint64_t>(col / packedShape[2])};
        auto byteOffset = getLocalPrintTensorByteOffset(packedCoord, packedStride, elementBytes);
        if (!byteOffset)
            return emitOpError("requires an L1 byte offset representable as uint32");
        if (std::gcd<uint64_t>(ptr.getAlignment(), *byteOffset) < 32)
            return emitOpError("requires a statically proven 32-byte aligned L1 address");
    }
    if (ptr.getAddrspace() == AddressSpace::l0c) {
        if (!hasEnclosingRegion<CubeOp>(getOperation()))
            return emitOpError("requires L0C tensors to be nested in a tla.cube region");
        if (tensorType.getLayout().getLayoutTag() != LayoutTag::L0Clayout)
            return emitOpError("requires an L0C tensor with L0Clayout");
        llvm::SmallVector<int64_t, 4> originShape;
        llvm::SmallVector<int64_t, 4> packedShape;
        llvm::SmallVector<int64_t, 4> packedStride;
        llvm::SmallVector<int64_t, 4> coords;
        if (failed(getIndexTreeLeavesForVerify(
                getOperation(), tensorType.getLayout().getOrigin(), originShape, "tensor origin shape")) ||
            failed(getIndexTreeLeavesForVerify(
                getOperation(), tensorType.getLayout().getShape(), packedShape, "tensor packed shape")) ||
            failed(getIndexTreeLeavesForVerify(
                getOperation(), tensorType.getLayout().getStride(), packedStride, "tensor packed stride")) ||
            failed(getIndexTreeLeavesForVerify(getOperation(), tensorType.getCoord(), coords, "tensor coordinates")))
            return mlir::failure();
        if (originShape.size() != 2 || packedShape.size() != 4 || packedStride.size() != 4 || coords.size() != 2)
            return emitOpError("requires a static rank-2 L0C view");
        int64_t rows = originShape[0];
        int64_t cols = originShape[1];
        int64_t row = coords[0];
        int64_t col = coords[1];
        if (rows < 16 || cols < 16 || row < 0 || col < 0 ||
            llvm::any_of(packedShape, [](int64_t value) { return value <= 0; }) ||
            llvm::any_of(packedStride, [](int64_t value) { return value < 0; }))
            return emitOpError("requires a complete 16x16 L0C tile inside the logical shape");
        if (row % 16 != 0 || col % 16 != 0)
            return emitOpError("requires L0C coordinates aligned to a 16x16 fractal");
        std::array<uint64_t, 4> packedCoord = {
            static_cast<uint64_t>(row % packedShape[0]), static_cast<uint64_t>(row / packedShape[0]),
            static_cast<uint64_t>(col % packedShape[2]), static_cast<uint64_t>(col / packedShape[2])};
        uint64_t elementBytes = elementType.getIntOrFloatBitWidth() / 8;
        auto byteOffset = getLocalPrintTensorByteOffset(packedCoord, packedStride, elementBytes);
        if (!byteOffset)
            return emitOpError("requires an L0C byte offset representable as uint32");
        if (std::gcd<uint64_t>(ptr.getAlignment(), *byteOffset) < 32)
            return emitOpError("requires a statically proven 32-byte aligned L0C address");
    }
    if (getShape().size() < 1 || getShape().size() > 2)
        return emitOpError("shape must have rank 1 or 2");
    llvm::SmallVector<int64_t, 4> tensorShape;
    if (ptr.getAddrspace() == AddressSpace::l0c) {
        tensorShape.assign({16, 16});
    } else {
        auto shapeType = ptr.getAddrspace() == AddressSpace::l1 ? tensorType.getLayout().getOrigin() :
                                                                  tensorType.getLayout().getShape();
        if (failed(getIndexTreeLeavesForVerify(getOperation(), shapeType, tensorShape, "tensor shape")))
            return mlir::failure();
        if (tensorShape.size() > 2) {
            tensorShape.clear();
            if (failed(getIndexTreeLeavesForVerify(
                    getOperation(), tensorType.getLayout().getOrigin(), tensorShape, "tensor logical shape")))
                return mlir::failure();
        }
    }
    if (getShape().size() != tensorShape.size())
        return emitOpError("shape must match the logical tensor shape");
    for (auto [declared, actual] : llvm::zip_equal(getShape(), tensorShape)) {
        if (declared == -1 && mlir::ShapedType::isDynamic(actual))
            continue;
        if (declared != actual)
            return emitOpError("shape must match the logical tensor shape");
    }
    std::optional<int64_t> product = 1;
    for (int64_t extent : getShape()) {
        if (extent == -1) {
            product.reset();
            continue;
        }
        if (extent < 1 || (product && *product > std::numeric_limits<int64_t>::max() / extent))
            return emitOpError("shape must contain a valid positive element count");
        if (product)
            *product *= extent;
    }
    if (auto constant = getLength().getDefiningOp<mlir::arith::ConstantOp>()) {
        auto lengthAttr = mlir::dyn_cast<mlir::IntegerAttr>(constant.getValue());
        if (!lengthAttr)
            return emitOpError("constant length must be an integer");
        int64_t length = lengthAttr.getInt();
        int64_t maxLength = ptr.getAddrspace() == AddressSpace::l1  ? 8 :
                            ptr.getAddrspace() == AddressSpace::l0c ? 256 :
                                                                      kMaxFloat32Elements;
        if (auto vf = getOperation()->getParentOfType<VecFuncOp>()) {
            auto mode = vf->getAttrOfType<mlir::StringAttr>("mode");
            if (!mode || !mode.getValue().equals_insensitive("simt"))
                maxLength = 1952 * 8 / elementType.getIntOrFloatBitWidth();
        }
        if (length < 1 || length > maxLength)
            return emitOpError() << "length must be between 1 and " << maxLength << " elements";
        if (product && length > *product)
            return emitOpError("length must not exceed the tensor element count");
        if (ptr.getAddrspace() == AddressSpace::l0c && length != 256)
            return emitOpError("L0C tensor printing requires exactly 256 elements");
    }
    return mlir::success();
}

static mlir::ParseResult parseIndexTreeValueOp(mlir::OpAsmParser& parser, mlir::OperationState& result)
{
    llvm::SmallVector<mlir::OpAsmParser::UnresolvedOperand, 4> dynElemsOperands;
    llvm::SMLoc dynElemsOperandsLoc = parser.getCurrentLocation();
    mlir::Type resultType;

    if (parser.parseOperandList(dynElemsOperands))
        return mlir::failure();
    if (parser.parseOptionalAttrDict(result.attributes))
        return mlir::failure();
    if (parser.parseArrow())
        return mlir::failure();
    if (parser.parseType(resultType))
        return mlir::failure();

    result.addTypes(resultType);
    if (parser.resolveOperands(
            dynElemsOperands, parser.getBuilder().getIndexType(), dynElemsOperandsLoc, result.operands))
        return mlir::failure();
    return mlir::success();
}

template <typename OpTy>
static void printIndexTreeValueOp(OpTy op, mlir::OpAsmPrinter& printer)
{
    if (!op.getDynElems().empty()) {
        printer << ' ';
        printer << op.getDynElems();
    }
    printer.printOptionalAttrDict(op->getAttrs());
    printer << " -> ";
    printer.printType(op.getResult().getType());
}

mlir::ParseResult DebugPrintOp::parse(mlir::OpAsmParser& parser, mlir::OperationState& result)
{
    llvm::SmallVector<mlir::OpAsmParser::UnresolvedOperand, 4> operands;
    llvm::SmallVector<mlir::Type, 4> operandTypes;
    llvm::SMLoc operandsLoc = parser.getCurrentLocation();

    if (succeeded(parser.parseOptionalKeyword("format"))) {
        mlir::StringAttr formatAttr;
        if (parser.parseAttribute(formatAttr, "format", result.attributes))
            return mlir::failure();
        if (succeeded(parser.parseOptionalLParen())) {
            if (failed(parser.parseOptionalRParen())) {
                do {
                    mlir::OpAsmParser::UnresolvedOperand operand;
                    if (parser.parseOperand(operand))
                        return mlir::failure();
                    operands.push_back(operand);
                } while (succeeded(parser.parseOptionalComma()));
                if (parser.parseRParen())
                    return mlir::failure();
            }
        }
        if (parser.parseOptionalAttrDict(result.attributes) || parser.parseColon() || parser.parseLParen())
            return mlir::failure();
        if (failed(parser.parseOptionalRParen())) {
            do {
                mlir::Type type;
                if (parser.parseType(type))
                    return mlir::failure();
                operandTypes.push_back(type);
            } while (succeeded(parser.parseOptionalComma()));
            if (parser.parseRParen())
                return mlir::failure();
        }
        if (operands.size() != operandTypes.size())
            return parser.emitError(operandsLoc)
                   << "expected " << operands.size() << " operand types, got " << operandTypes.size();
        if (parser.resolveOperands(operands, operandTypes, operandsLoc, result.operands))
            return mlir::failure();
        return mlir::success();
    }

    mlir::OpAsmParser::UnresolvedOperand operand;
    mlir::Type type;
    operandsLoc = parser.getCurrentLocation();
    if (parser.parseOperand(operand) || parser.parseOptionalAttrDict(result.attributes) || parser.parseColon() ||
        parser.parseType(type))
        return mlir::failure();
    if (parser.resolveOperand(operand, type, result.operands))
        return mlir::failure();
    return mlir::success();
}

void DebugPrintOp::print(mlir::OpAsmPrinter& printer)
{
    auto formatAttr = (*this)->getAttrOfType<mlir::StringAttr>("format");
    if (!formatAttr) {
        printer << ' ' << getValues().front();
        printer.printOptionalAttrDict((*this)->getAttrs());
        printer << " : ";
        printer.printType(getValues().front().getType());
        return;
    }

    printer << " format " << formatAttr;
    printer << '(' << getValues() << ')';
    printer.printOptionalAttrDict((*this)->getAttrs(), {"format"});
    printer << " : (";
    for (auto item : llvm::enumerate(getValues())) {
        if (item.index() != 0)
            printer << ", ";
        printer.printType(item.value().getType());
    }
    printer << ')';
}

mlir::ParseResult MakeShapeOp::parse(mlir::OpAsmParser& parser, mlir::OperationState& result)
{
    return parseIndexTreeValueOp(parser, result);
}

void MakeShapeOp::print(mlir::OpAsmPrinter& printer)
{
    printIndexTreeValueOp(*this, printer);
}

mlir::ParseResult MakeCoordOp::parse(mlir::OpAsmParser& parser, mlir::OperationState& result)
{
    return parseIndexTreeValueOp(parser, result);
}

void MakeCoordOp::print(mlir::OpAsmPrinter& printer)
{
    printIndexTreeValueOp(*this, printer);
}

mlir::ParseResult MakeStrideOp::parse(mlir::OpAsmParser& parser, mlir::OperationState& result)
{
    return parseIndexTreeValueOp(parser, result);
}

void MakeStrideOp::print(mlir::OpAsmPrinter& printer)
{
    printIndexTreeValueOp(*this, printer);
}

mlir::ParseResult FuncOp::parse(mlir::OpAsmParser& parser, mlir::OperationState& result)
{
    mlir::StringAttr symNameAttr;
    if (parser.parseSymbolName(symNameAttr, "sym_name", result.attributes))
        return llvm::failure();

    llvm::SmallVector<mlir::OpAsmParser::Argument, 4> arguments;
    if (parser.parseArgumentList(
            arguments, mlir::OpAsmParser::Delimiter::Paren,
            /*allowType=*/true))
        return llvm::failure();

    llvm::SmallVector<mlir::Type, 4> inputTypes;
    inputTypes.reserve(arguments.size());
    for (const auto& arg : arguments) {
        if (!arg.type)
            return parser.emitError(arg.ssaName.location) << "expected type for function argument";
        inputTypes.push_back(arg.type);
    }

    llvm::SmallVector<mlir::Type, 2> resultTypes;
    if (parser.parseOptionalArrowTypeList(resultTypes))
        return llvm::failure();

    if (parser.parseOptionalAttrDictWithKeyword(result.attributes))
        return llvm::failure();

    auto functionType = parser.getBuilder().getFunctionType(inputTypes, resultTypes);
    result.attributes.set("function_type", mlir::TypeAttr::get(functionType));

    mlir::Region* body = result.addRegion();
    if (parser.parseRegion(*body, arguments))
        return llvm::failure();

    return llvm::success();
}

void FuncOp::print(mlir::OpAsmPrinter& printer)
{
    printer << " @" << getSymName() << "(";

    auto fnType = llvm::dyn_cast<mlir::FunctionType>(getFunctionType());
    llvm::ArrayRef<mlir::Type> inputs = fnType ? fnType.getInputs() : llvm::ArrayRef<mlir::Type>{};
    mlir::Block* entry = getBody().empty() ? nullptr : &getBody().front();

    for (size_t i = 0; i < inputs.size(); ++i) {
        if (i)
            printer << ", ";
        if (entry && i < entry->getNumArguments()) {
            printer << entry->getArgument(i);
        } else {
            printer << "%arg" << i;
        }
        printer << ": " << inputs[i];
    }
    printer << ")";

    if (fnType && !fnType.getResults().empty()) {
        printer << " -> (";
        for (size_t i = 0; i < fnType.getResults().size(); ++i) {
            if (i)
                printer << ", ";
            printer.printType(fnType.getResults()[i]);
        }
        printer << ")";
    }

    printer.printOptionalAttrDictWithKeyword((*this)->getAttrs(), {"sym_name", "function_type"});
    printer << " ";
    printer.printRegion(getBody(), /*printEntryBlockArgs=*/false);
}

static mlir::LogicalResult verifyScalarTensorAccess(
    mlir::Operation* op, TlaTensorType tensorType, mlir::ValueRange indices, mlir::Type scalarType,
    llvm::StringRef tensorRole)
{
    PtrType ptrType = tensorType.getPtr();
    AddressSpace addrspace = ptrType.getAddrspace();
    if (addrspace != AddressSpace::gm && addrspace != AddressSpace::ub)
        return op->emitOpError() << tensorRole << " must be in gm or ub address space";

    if (addrspace == AddressSpace::ub) {
        if (!hasEnclosing<VectorOp>(op))
            return op->emitOpError("UB scalar access must be nested inside a tla.vector region");
    }

    auto layoutTag = tensorType.getLayout().getLayoutTag();
    if (!isLinearLayout(layoutTag))
        return op->emitOpError() << tensorRole << " layout must be RowMajor, ColumnMajor, or VectorLayout";

    mlir::Type expected = ptrType.getPointee();
    if (scalarType != expected)
        return op->emitOpError("scalar type must match tensor element type, expected ")
               << expected << ", got " << scalarType;

    llvm::SmallVector<int64_t, 4> shapeLeaves;
    if (failed(getIndexTreeLeavesForVerify(op, tensorType.getLayout().getShape(), shapeLeaves, "shape")))
        return mlir::failure();
    size_t rank = shapeLeaves.size();
    if (rank != 1 && rank != 2)
        return op->emitOpError("scalar access requires a rank-1 or rank-2 tensor view");
    if (indices.size() != rank)
        return op->emitOpError() << "scalar access index count must match tensor logical rank " << rank;
    for (mlir::Value index : indices) {
        if (!index.getType().isIndex())
            return op->emitOpError("scalar access indices must be index-typed");
    }

    if (addrspace != AddressSpace::ub)
        return mlir::success();
    ShapeType origin = tensorType.getLayout().getOrigin();
    if (!origin)
        return op->emitOpError("UB scalar access requires tensor origin shape metadata");
    llvm::SmallVector<int64_t, 4> originShape;
    if (failed(getIndexTreeLeavesForVerify(op, origin, originShape, "origin shape")))
        return mlir::failure();
    if (originShape.size() != rank)
        return op->emitOpError("UB scalar access origin rank must match tensor rank");
    for (auto [index, length] : llvm::zip_equal(indices, originShape)) {
        if (length != mlir::ShapedType::kDynamic && length <= 0)
            return op->emitOpError("UB scalar access requires a non-empty tensor view");
        if (failed(verifyConstantScalarIndexInBounds(op, index, length)))
            return mlir::failure();
    }
    return mlir::success();
}

mlir::LogicalResult ScalarLoadOp::verify()
{
    return verifyScalarTensorAccess(
        getOperation(), getSource().getType(), getIndices(), getResult().getType(), "source !tla.tensor");
}

mlir::LogicalResult ScalarStoreOp::verify()
{
    return verifyScalarTensorAccess(
        getOperation(), getDest().getType(), getIndices(), getValue().getType(), "destination !tla.tensor");
}

} // namespace tla
