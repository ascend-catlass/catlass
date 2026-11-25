#include "Dialect/Tla/IR/TlaOps.h"

#include <array>
#include <limits>
#include <numeric>
#include <optional>
#include <utility>

#include "Dialect/Tla/IR/TlaAttrs.h"
#include "Dialect/Tla/IR/TlaTypes.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/OpImplementation.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Support/raw_ostream.h"

#define GET_OP_CLASSES
#include "tla/Ops.cpp.inc"

namespace tla {

mlir::FailureOr<mlir::Value> resolveTensorBacking(mlir::Value tensor)
{
    if (mlir::isa<mlir::MemRefType, PtrType>(tensor.getType()))
        return tensor;
    if (auto desc = tensor.getDefiningOp<TensorDescOp>())
        return resolveTensorBacking(desc.getBase());
    if (auto tileView = tensor.getDefiningOp<TileViewOp>())
        return resolveTensorBacking(tileView.getSource());
    if (auto makeTensor = tensor.getDefiningOp<MakeTensorOp>())
        return makeTensor.getPtr();
    if (auto makeTensorLike = tensor.getDefiningOp<MakeTensorLikeOp>())
        return makeTensorLike.getPtr();
    return mlir::failure();
}

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
    if (elementType.isF16() || elementType.isBF16() || elementType.isF32())
        return true;
    auto intType = mlir::dyn_cast<mlir::IntegerType>(elementType);
    if (!intType)
        return false;
    unsigned width = intType.getWidth();
    // AscendC Compare/Compares accept the 8/16/32/64-bit integer families.
    // Unsigned elements keep VCmpU* selection; signed and signless share the
    // signed instruction family.
    return width == 8 || width == 16 || width == 32 || width == 64;
}

// AscendC Select accepts bool plus the 8/16/32/64-bit widths. AVE vectors are
// signless; Bool rides the b8 carrier, so it is accepted as i1 here. The
// low-precision formats select their byte/pair carriers: f8e4m3fn/f8e5m2 use
// the native f8 vsel, hif8 and packed FP4 ride the i8 path.
static bool isSupportedSelectElementType(mlir::Type elementType)
{
    if (elementType.isInteger(1))
        return true;
    if (elementType.isF16() || elementType.isBF16() || elementType.isF32())
        return true;
    if (mlir::isa<
            mlir::Float8E4M3FNType, mlir::Float8E5M2Type, ::tla::HiFloat8Type, ::tla::Float4E2M1x2Type,
            ::tla::Float4E1M2x2Type>(elementType))
        return true;
    auto intType = mlir::dyn_cast<mlir::IntegerType>(elementType);
    if (!intType)
        return false;
    unsigned width = intType.getWidth();
    return width == 8 || width == 16 || width == 32 || width == 64;
}

// The scalar carrier of a vector-scalar compare is signless: arith.constant
// and the AVE scalar ABI only accept signless integers. An unsigned vector
// element therefore accepts its same-width signless twin.
static bool isCompareScalarTypeCompatible(mlir::Type scalarType, mlir::Type elementType)
{
    if (scalarType == elementType)
        return true;
    auto elemInt = mlir::dyn_cast<mlir::IntegerType>(elementType);
    auto scalarInt = mlir::dyn_cast<mlir::IntegerType>(scalarType);
    return elemInt && scalarInt && elemInt.isUnsigned() && scalarInt.isSignless() &&
           elemInt.getWidth() == scalarInt.getWidth();
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

mlir::LogicalResult TensorPtrWithCoordOp::verify()
{
    auto tensorTy = mlir::dyn_cast<TlaTensorType>(getSrc().getType());
    if (!tensorTy)
        return emitOpError("src must be a !tla.tensor");
    auto resTy = mlir::dyn_cast<PtrType>(getPtr().getType());
    if (!resTy)
        return emitOpError("result must be !tla.ptr");
    if (tensorTy.getPtr() != resTy)
        return emitOpError("result ptr type must match the tensor's embedded pointer type");
    if (!isLinearLayout(tensorTy.getLayout().getLayoutTag()))
        return emitOpError(
            "src must use a linear layout (RowMajor, ColumnMajor or VectorLayout); "
            "the view-start offset formula does not cover packed layouts");
    llvm::SmallVector<int64_t, 2> coordLeaves;
    if (failed(getIndexTreeLeavesForVerify(getOperation(), tensorTy.getCoord(), coordLeaves, "coordinate")))
        return mlir::failure();
    if (coordLeaves.size() != 1 && coordLeaves.size() != 2)
        return emitOpError(
            "src coordinate must have one or two leaves; the view-start offset "
            "formula covers exactly two logical dimensions");
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
    bool batched = originLeaves.size() == 3;
    if (getBatchInfo().size() != (batched ? 4u : 0u))
        return emitOpError("batch metadata requires leading shape, stride, origin and coordinate");
    if (batched) {
        if (shapeLeaves.empty() || strideLeaves.empty() || coordLeaves.size() != 3)
            return emitOpError("malformed batch-matrix metadata");
        shapeLeaves.erase(shapeLeaves.begin());
        strideLeaves.erase(strideLeaves.begin());
        originLeaves.erase(originLeaves.begin());
        coordLeaves.erase(coordLeaves.begin());
    }
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

// Batched descriptors are currently consumed only by GM-to-L1 copy. Other
// tile operations must reject them rather than silently operating on matrix 0.
static mlir::LogicalResult requireMatrixTiles(mlir::Operation* op)
{
    for (mlir::Value value : op->getOperands()) {
        auto tensor = mlir::dyn_cast<TlaTensorType>(value.getType());
        if (!tensor)
            continue;
        llvm::SmallVector<int64_t, 4> origin;
        if (mlir::failed(getTlaIndexTreeLeaves(tensor.getLayout().getOrigin().getTree(), origin)))
            return mlir::failure();
        if (origin.size() == 3)
            return op->emitOpError("currently requires individual matrix tiles; batched tensors support GM-to-L1 copy");
    }
    return mlir::success();
}

mlir::LogicalResult MmadOp::verify()
{
    if (mlir::failed(requireMatrixTiles(getOperation())))
        return mlir::failure();
    if (!hasEnclosingRegion<CubeOp>(getOperation()))
        return emitOpError("must be nested inside a tla.cube region");

    // The ODS operand constraints already guarantee TlaTensorType here.
    auto accTy = mlir::cast<TlaTensorType>(getAcc().getType());
    auto lhsTy = mlir::cast<TlaTensorType>(getLhs().getType());
    auto rhsTy = mlir::cast<TlaTensorType>(getRhs().getType());

    LayoutTag accTag = accTy.getLayout().getLayoutTag();
    if (accTag != LayoutTag::L0Clayout)
        return emitOpError() << "acc must be tagged L0Clayout, got " << stringifyLayoutTag(accTag);

    LayoutTag rhsTag = rhsTy.getLayout().getLayoutTag();
    if (rhsTag != LayoutTag::nZ)
        return emitOpError() << "mmad requires an nZ rhs tile, got " << stringifyLayoutTag(rhsTag);

    LayoutTag lhsTag = lhsTy.getLayout().getLayoutTag();
    if (lhsTag != LayoutTag::zN && lhsTag != LayoutTag::VectorLayout)
        return emitOpError() << "lhs must be tagged zN (matrix form) or VectorLayout (gemv form), got "
                             << stringifyLayoutTag(lhsTag);

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

        if (failed(checkF32SourceOperand(lhsTy, "lhs")))
            return mlir::failure();
        if (failed(checkF32SourceOperand(rhsTy, "rhs")))
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
    if (mlir::failed(requireMatrixTiles(getOperation())))
        return mlir::failure();
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
        // An fp4 ML0 buffer is addressed in 512-byte units, like any other
        // L0 tile, so the 512 the frontend enforces for l0a / l0b is enough.
    } else {
        if (lhsTag != LayoutTag::zN || rhsTag != LayoutTag::nZ)
            return emitOpError("fp8 MX operands must be tagged zN (lhs) and nZ (rhs)");
        if (!isFp8(lhsTy.getPtr().getPointee()) || !isFp8(rhsTy.getPtr().getPointee()))
            return emitOpError("MX operands must be f8E4M3FN / f8E5M2, or fp4-tagged i8 tiles");

        // mxfp8 raises the L0 alignment to 1024: the instruction reads the
        // operand as a whole 1024-byte block, so a buffer placed on a 512-byte
        // boundary is half a block off. The frontend cannot enforce this --
        // tla.allocate sees only the address space, and 512 is right for every
        // other L0 user -- so the requirement is checked here, where the
        // element format is known. fp4 above stays at 512.
        //
        // The test is divisibility, not size: an alignment of N promises only
        // that the address is a multiple of N, so it promises 1024-byte
        // alignment exactly when 1024 divides N. 1536 is a legal L0A alignment
        // (3 x 512) yet not a multiple of 1024, and an address it produces can
        // land 512 bytes into an mxfp8 block -- so "at least 1024" would let it
        // through.
        constexpr uint64_t kMxL0AlignBytes = 1024;
        for (auto [ptr, side] : {std::pair{lhsTy.getPtr(), "lhs"}, std::pair{rhsTy.getPtr(), "rhs"}}) {
            if (ptr.getAlignment() % kMxL0AlignBytes != 0)
                return emitOpError() << side << " must be " << kMxL0AlignBytes
                                     << "-byte aligned for an mxfp8 MX operand, but the tile is aligned to "
                                     << ptr.getAlignment()
                                     << " bytes. Allocate it with byte_alignment=" << kMxL0AlignBytes;
        }
    }
    return mlir::success();
}

mlir::LogicalResult CopyMxOp::verify()
{
    if (mlir::failed(requireMatrixTiles(getOperation())))
        return mlir::failure();
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
    auto dstLayout = dstTy.getLayout().getLayoutTag();
    bool toL0B = dst == AddressSpace::l0b;
    if (!((dst == AddressSpace::l0a && dstLayout == LayoutTag::zN) || (toL0B && dstLayout == LayoutTag::nZ)))
        return emitOpError("img2col destination must be an l0a tile tagged zN or an l0b tile tagged nZ");
    if (srcTy.getPtr().getPointee() != dstTy.getPtr().getPointee())
        return emitOpError("img2col source and destination element types must match");
    ::mlir::Type elem = srcTy.getPtr().getPointee();
    int64_t elementBytes = getByteSizeOfFixedWidthScalarType(elem);
    if (!isImg2colElementType(elem) || (toL0B && elementBytes == 1))
        return emitOpError("unsupported img2col element type for ") << (toL0B ? "l0b: " : "l0a: ") << elem;

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
        if (mlir::failed(verifyConstantRange(value, name, 1, 32767)))
            return mlir::failure();
    }
    for (auto [value, name] : {std::pair{getStrideH(), "stride_h"}, std::pair{getStrideW(), "stride_w"}}) {
        if (mlir::failed(verifyConstantRange(value, name, 1, 63)))
            return mlir::failure();
    }
    for (auto [value, name] : {std::pair{getDilationH(), "dilation_h"}, std::pair{getDilationW(), "dilation_w"}}) {
        if (mlir::failed(verifyConstantRange(value, name, 1, 255)))
            return mlir::failure();
    }
    for (auto [value, name] :
         {std::pair{getPadLeft(), "pad_left"}, std::pair{getPadRight(), "pad_right"}, std::pair{getPadTop(), "pad_top"},
          std::pair{getPadBottom(), "pad_bottom"}}) {
        if (mlir::failed(verifyConstantRange(value, name, 0, 255)))
            return mlir::failure();
    }

    llvm::SmallVector<int64_t, 4> srcShape, dstShape, srcStride, dstStride;
    llvm::SmallVector<int64_t, 2> srcCoord, dstCoord, dstOrigin;
    if (failed(getTlaIndexTreeLeaves(srcTy.getLayout().getShape().getTree(), srcShape)) ||
        failed(getTlaIndexTreeLeaves(dstTy.getLayout().getShape().getTree(), dstShape)) ||
        failed(getTlaIndexTreeLeaves(srcTy.getLayout().getStride().getTree(), srcStride)) ||
        failed(getTlaIndexTreeLeaves(dstTy.getLayout().getStride().getTree(), dstStride)) || srcShape.size() != 4 ||
        dstShape.size() != 4 || srcStride.size() != 4 || dstStride.size() != 4)
        return emitOpError("requires four-leaf source and destination shapes and strides");
    if (!dstTy.getLayout().getOrigin() ||
        failed(getTlaIndexTreeLeaves(dstTy.getLayout().getOrigin().getTree(), dstOrigin)) ||
        failed(getTlaIndexTreeLeaves(srcTy.getCoord().getTree(), srcCoord)) ||
        failed(getTlaIndexTreeLeaves(dstTy.getCoord().getTree(), dstCoord)) || srcCoord.size() != 2 ||
        dstCoord.size() != 2 || dstOrigin.size() != 2)
        return emitOpError("requires two-axis coordinates and destination origin shape");
    int64_t c0 = 32 / elementBytes;
    if (srcShape[2] != c0 || srcShape[3] < 1 || srcShape[3] > 65535 / c0)
        return emitOpError("source channels must be static whole C1 slices within 1..65535 channels");

    llvm::APInt fmapH, fmapW;
    if (mlir::matchPattern(getFmapH(), mlir::m_ConstantInt(&fmapH)) &&
        mlir::matchPattern(getFmapW(), mlir::m_ConstantInt(&fmapW))) {
        int64_t positions = fmapH.getSExtValue() * fmapW.getSExtValue();
        if (!mlir::ShapedType::isDynamic(srcShape[0]) && !mlir::ShapedType::isDynamic(srcShape[1]) &&
            positions > srcShape[0] * srcShape[1])
            return emitOpError("fmap_hw exceeds the number of spatial positions in the source view");
        if (srcShape[3] > 1 && !mlir::ShapedType::isDynamic(srcStride[3]) && srcStride[3] != positions * c0)
            return emitOpError("source C1 stride must equal fmap_h * fmap_w * C0 for multiple C1 slices");
    }

    if (!mlir::ShapedType::isDynamic(dstCoord[0]) && (dstCoord[0] < 0 || dstCoord[0] > 32767))
        return emitOpError("destination M coordinate must be within 0..32767");
    if (!mlir::ShapedType::isDynamic(dstShape[0]) && !mlir::ShapedType::isDynamic(dstShape[1]) &&
        dstShape[0] * dstShape[1] > 65535)
        return emitOpError("destination M extent must be within 0..65535");

    int64_t dstKBlock = toL0B ? 16 : c0;
    int64_t kBlocks = dstShape[3];
    if (dstShape[2] != dstKBlock || kBlocks < 1)
        return emitOpError("destination storage must have static ") << dstKBlock << "-element K blocks";
    int64_t logicalK = dstOrigin[1];
    if (logicalK < 1 || logicalK > (65535 / c0) * c0)
        return emitOpError("destination logical K extent must be static and fit a C0-rounded length in 1..65535");
    int64_t kExtent = ((logicalK + c0 - 1) / c0) * c0;
    if ((kExtent + dstKBlock - 1) / dstKBlock > kBlocks)
        return emitOpError("destination K storage is smaller than the K transfer length");
    int64_t kStart = dstCoord[1];
    if (kStart < 0 || kStart > 65535 || kStart % dstKBlock != 0)
        return emitOpError("destination K coordinate must be static, within 0..65535 and aligned to ")
               << dstKBlock << " elements";
    int64_t outerStride = dstStride[toL0B ? 1 : 3];
    int64_t elementsPerFractal = 16 * c0;
    if (outerStride != mlir::ShapedType::kDynamic &&
        (outerStride <= 0 || outerStride % elementsPerFractal != 0 || outerStride / elementsPerFractal > 65535))
        return emitOpError("destination outer stride must fit the unsigned 16-bit 512-byte-fractal field");

    int64_t repeatStride = getRepeatStride();
    int64_t repeatTime = getRepeatTime();
    int64_t repeatMode = getRepeatMode();
    if (repeatStride < 0 || repeatStride > 65535)
        return emitOpError("repeat_stride must be in 0..65535");
    if (repeatTime < 0 || repeatTime > 255)
        return emitOpError("repeat_time must be in 0..255");
    if (repeatMode != 0 && repeatMode != 1)
        return emitOpError("repeat_mode must be 0 (height/M) or 1 (width/K)");
    if (repeatTime == 0)
        return mlir::success();

    int64_t fullK = c0 * srcShape[3] * filterExtents[0] * filterExtents[1];
    int64_t kEnd = kStart + kExtent;
    if (repeatMode == 1)
        kEnd += (repeatTime - 1) * repeatStride * c0;
    if (kEnd > fullK)
        return emitOpError("destination K window exceeds the source channel/filter extent");

    // Every L0 outer-fractal stride is a multiple of 512 bytes. Only the
    // coordinates within the fractal can make an otherwise aligned base miss.
    if (!mlir::ShapedType::isDynamic(dstCoord[0])) {
        int64_t row = dstCoord[0];
        int64_t withinFractal = toL0B ? row % c0 + (kStart % 16) * c0 : (row % 16) * c0 + kStart % c0;
        int64_t baseAlignment = std::gcd<int64_t>(dstTy.getPtr().getAlignment(), 512);
        if ((withinFractal * elementBytes) % baseAlignment != 0)
            return emitOpError("effective L0 destination address must be 512-byte aligned");
    }

    // Each spatial row contributes one 32-byte C0. Only the channel offset
    // can change the alignment residue.
    if (srcCoord[1] >= 0) {
        int64_t c1Residue = (srcCoord[1] / c0) % c0;
        bool staticStride = !mlir::ShapedType::isDynamic(srcStride[3]);
        if (c1Residue == 0 || staticStride) {
            int64_t strideResidue = staticStride ? srcStride[3] % c0 : 0;
            int64_t withinBlock = (srcCoord[1] % c0 + c1Residue * strideResidue) % c0;
            int64_t baseAlignment = std::gcd<int64_t>(srcTy.getPtr().getAlignment(), 32);
            if ((withinBlock * elementBytes) % baseAlignment != 0)
                return emitOpError("effective L1 source address must be 32-byte aligned");
        }
    }
    return mlir::success();
}

mlir::LogicalResult FillOp::verify()
{
    if (mlir::failed(requireMatrixTiles(getOperation())))
        return mlir::failure();
    if (!hasEnclosingRegion<CubeOp>(getOperation()))
        return emitOpError("must be nested inside a tla.cube region");
    auto dstTy = mlir::dyn_cast<TlaTensorType>(getDst().getType());
    if (!dstTy)
        return mlir::success(); // Operand type verifier handles malformed tensors.

    if (dstTy.getPtr().getAddrspace() != AddressSpace::l1)
        return emitOpError("fill destination must be an l1 tile");
    LayoutTag tag = dstTy.getLayout().getLayoutTag();
    bool scaleLayout = tag == LayoutTag::zZMxScale || tag == LayoutTag::nNMxScale;
    if (tag != LayoutTag::zN && tag != LayoutTag::nZ && !scaleLayout)
        return emitOpError("fill destination must be tagged zN or nZ, or zZMxScale/nNMxScale for E8M0");
    ::mlir::Type elem = dstTy.getPtr().getPointee();
    bool scaleElement = mlir::isa<::tla::Float8E8M0Type>(elem);
    bool packedElement = ::tla::isPackedFp4Type(elem) || mlir::isa<mlir::Float8E4M3FNType, mlir::Float8E5M2Type>(elem);
    if (!::tla::isOfficialL1FillElementType(elem) && !packedElement && !scaleElement)
        return emitOpError("unsupported L1 fill destination element type");
    if (scaleLayout != scaleElement)
        return emitOpError("E8M0 fill requires zZMxScale/nNMxScale; operand fill requires zN/nZ");
    if (packedElement || scaleElement) {
        ::llvm::APInt valueBits;
        if (mlir::matchPattern(getValue(), mlir::m_ConstantInt(&valueBits)) && !valueBits.isIntN(16))
            return emitOpError("fill bit pattern must be an unsigned 16-bit value");
    }
    return mlir::success();
}
// One thread block may hold at most this many threads on the supported targets.
// The lowering packs the product into hivm_regbaseintrins::SIMT_EntryAttr, whose
// value is a uint32_t, so an unchecked product would also truncate.
static constexpr int64_t kMaxSimtThreadsPerBlock = 2048;
// A warp is this many lanes, which is also the width of the shuffle control
// field's index: the hardware keeps its low 5 bits and discards the rest.
static constexpr int64_t kWarpSize = 32;

mlir::LogicalResult ArchLoadBypassDcacheOp::verify()
{
    auto* op = getOperation();
    bool inCube = hasEnclosing<CubeOp>(op);
    bool inVector = hasEnclosing<VectorOp>(op);
    if (!inCube && !inVector && !isInLoweredFunc(op))
        return op->emitOpError("must be nested inside a tla.cube or tla.vector region");
    if (inCube && inVector)
        return op->emitOpError("mixed cube/vector placement is unsupported");
    if (hasEnclosing<VecFuncOp>(op))
        return op->emitOpError("requires scalar core code outside tla.vec.func");
    auto resultType = mlir::cast<mlir::IntegerType>(getResult().getType());
    if (!llvm::is_contained(llvm::ArrayRef<unsigned>{8, 16, 32, 64}, resultType.getWidth()))
        return emitOpError("result must be an 8-, 16-, 32-, or 64-bit integer");
    return mlir::success();
}

mlir::LogicalResult ArchDcachePrefetchOp::verify()
{
    if (mlir::cast<PtrType>(getAddress().getType()).getAddrspace() != AddressSpace::gm)
        return emitOpError("requires a GM pointer");
    if ((*this)->getParentOfType<VecFuncOp>() ||
        (!(*this)->getParentOfType<VectorOp>() && !(*this)->getParentOfType<CubeOp>()))
        return emitOpError("requires a scalar tla.vector or tla.cube region");
    return mlir::success();
}

mlir::LogicalResult ArchDcciCachelinesOp::verify()
{
    if (getTarget() == ::DcciTarget::ub)
        return emitOpError(
            "target 'ub' is not available: the native whole-UB form is AIV-only and "
            "requires UB dcache enabled through CTRL[49], which no TLA operation sets");
    return mlir::success();
}

// Before core-region flattening, the enclosing region identifies the issuer.
// After flattening, the split function carries the concrete core type.
static int64_t getScalarICacheUnitLimit(mlir::Operation* op)
{
    if (op->getParentOfType<VecFuncOp>())
        return 0;
    bool aiv = bool(op->getParentOfType<VectorOp>());
    bool aic = bool(op->getParentOfType<CubeOp>());
    if (aiv || aic)
        return aiv == aic ? 0 : (aiv ? 7 : 15);
    if (auto func = op->getParentOfType<mlir::func::FuncOp>()) {
        auto issuer = func->getAttrOfType<mlir::hivm::TFuncCoreTypeAttr>(mlir::hivm::TFuncCoreTypeAttr::name);
        if (issuer && issuer.getFuncCoreType() == mlir::hivm::TFuncCoreType::AIV)
            return 7;
        if (issuer && issuer.getFuncCoreType() == mlir::hivm::TFuncCoreType::AIC)
            return 15;
    }
    return 0;
}

mlir::LogicalResult ArchICachePrefetchOp::verify()
{
    int64_t limit = getScalarICacheUnitLimit(getOperation());
    if (!limit)
        return emitOpError("requires one scalar vector or cube region outside tla.vec.func");
    int64_t units = getPrefetchUnitsAttr().getInt();
    if (units < 1 || units > limit)
        return emitOpError() << "prefetch_units must be in [1, " << limit << "]";
    return mlir::success();
}

mlir::LogicalResult ArchGetICachePrefetchStatusOp::verify()
{
    if (!getScalarICacheUnitLimit(getOperation()))
        return emitOpError("requires one scalar vector or cube region outside tla.vec.func");
    return mlir::success();
}

mlir::LogicalResult ArchLaneIdxOp::verify()
{
    auto region = getOperation()->getParentOfType<VecFuncOp>();
    auto mode = region ? region->getAttrOfType<mlir::StringAttr>("mode") : mlir::StringAttr{};
    if (!mode || !mode.getValue().equals_insensitive("simt"))
        return emitOpError("must be nested inside a tla.vec.func with mode='simt'");
    return mlir::success();
}

mlir::LogicalResult ArchActivemaskOp::verify()
{
    auto region = getOperation()->getParentOfType<VecFuncOp>();
    auto mode = region ? region->getAttrOfType<mlir::StringAttr>("mode") : mlir::StringAttr{};
    if (!mode || !mode.getValue().equals_insensitive("simt"))
        return emitOpError("must be nested inside a tla.vec.func with mode='simt'");
    return mlir::success();
}

mlir::LogicalResult ArchLanemaskOp::verify()
{
    auto region = getOperation()->getParentOfType<VecFuncOp>();
    auto regionMode = region ? region->getAttrOfType<mlir::StringAttr>("mode") : mlir::StringAttr{};
    if (!regionMode || !regionMode.getValue().equals_insensitive("simt"))
        return emitOpError("must be nested inside a tla.vec.func with mode='simt'");
    auto mode = getMode();
    if (mode != "eq" && mode != "lt" && mode != "le" && mode != "gt" && mode != "ge")
        return emitOpError("mode must be one of eq, lt, le, gt, ge");
    return mlir::success();
}

mlir::LogicalResult ArchWarpReduceOp::verify()
{
    auto region = getOperation()->getParentOfType<VecFuncOp>();
    auto mode = region ? region->getAttrOfType<mlir::StringAttr>("mode") : mlir::StringAttr{};
    if (!mode || !mode.getValue().equals_insensitive("simt"))
        return emitOpError("must be nested inside a tla.vec.func with mode='simt'");
    if (getKind() != "add" && getKind() != "max" && getKind() != "min")
        return emitOpError("kind must be one of add, max, min");
    auto type = getValue().getType();
    bool integer = type.isSignlessInteger(8) || type.isSignlessInteger(16) || type.isSignlessInteger(32);
    if (!integer && !type.isF16() && !type.isF32())
        return emitOpError("value must be i8, i16, i32, f16, or f32");
    if (getIsUnsigned() && !integer)
        return emitOpError("isUnsigned requires an integer value");
    return mlir::success();
}

mlir::LogicalResult SubBlockNumOp::verify()
{
    if (hasEnclosing<VecFuncOp>(getOperation()))
        return emitOpError("compute outside tla.vec.func and pass the scalar into the region");
    return mlir::success();
}

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
    // 64-bit destinations are legal (i32/u32 sources widen to i64/u64); the
    // AIV stubs compose them because Ascend 950 has no native vunpack for
    // that width. 64-bit sources would need a 128-bit destination and stay
    // rejected.
    auto srcInt = mlir::dyn_cast<mlir::IntegerType>(src.getElementType());
    auto dstInt = mlir::dyn_cast<mlir::IntegerType>(dst.getElementType());
    if (!srcInt || !dstInt || srcInt.isUnsigned() != dstInt.isUnsigned() ||
        (srcInt.getWidth() != 8 && srcInt.getWidth() != 16 && srcInt.getWidth() != 32) ||
        dstInt.getWidth() != 2 * srcInt.getWidth())
        return emitOpError() << "unsupported vector element type pair " << src.getElementType() << " -> "
                             << dst.getElementType()
                             << "; expected 8->16, 16->32 or 32->64 integers with matching signedness";

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
    int64_t elementBytes = elementStorageBytes(vectorType.getElementType());
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

static mlir::LogicalResult verifyB32Pair(mlir::Operation* op, VectorSSAType lhs, VectorSSAType rhs, mlir::Value mask)
{
    auto vf = op->getParentOfType<VecFuncOp>();
    auto mode = vf ? vf->getAttrOfType<mlir::StringAttr>("mode") : mlir::StringAttr();
    if (!vf || (mode && !mode.getValue().equals_insensitive("simd")))
        return op->emitOpError("must be nested inside a SIMD tla.vec.func region");
    auto elem = mlir::dyn_cast<mlir::IntegerType>(lhs.getElementType());
    if (!elem || elem.isSigned() || elem.getWidth() != 32 || lhs.getValidLanes() != 64)
        return op->emitOpError("requires full 64-lane i32/u32 registers");
    if (lhs != rhs)
        return op->emitOpError("requires lhs and rhs to have the same VectorSSA type");
    return verifyMaskMatchesVector(op, mask, lhs);
}

template <typename CarryOp>
static mlir::LogicalResult verifyB32CarryPair(CarryOp op, mlir::Value dataResult)
{
    auto lhs = op.getLhs().getType();
    if (failed(verifyB32Pair(op.getOperation(), lhs, op.getRhs().getType(), op.getMask())) ||
        failed(verifyMaskMatchesVector(op.getOperation(), op.getCarryIn(), lhs)))
        return mlir::failure();
    if (dataResult.getType() != lhs || op.getCarryOut().getType() != op.getCarryIn().getType())
        return op.emitOpError("data result and carry-out types must match lhs and carry-in");
    return mlir::success();
}

mlir::LogicalResult AddCOp::verify()
{
    return verifyB32CarryPair(*this, getSum());
}

mlir::LogicalResult SubCOp::verify()
{
    return verifyB32CarryPair(*this, getDiff());
}

mlir::LogicalResult MullOp::verify()
{
    auto lhs = getLhs().getType();
    if (failed(verifyB32Pair(getOperation(), lhs, getRhs().getType(), getMask())))
        return mlir::failure();
    if (getLow().getType() != lhs || getHigh().getType() != lhs)
        return emitOpError("low and high result types must match lhs");
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

// AscendC Duplicate accepts Bool, the 8/16/32/64-bit integers, f16/bf16/f32,
// fp8 e4m3fn/e5m2, and the byte-carried TLA formats (hif8/e8m0/fp4x2).
static bool isSupportedFullElementType(mlir::Type elementType)
{
    if (auto intType = mlir::dyn_cast<mlir::IntegerType>(elementType)) {
        unsigned width = intType.getWidth();
        return width == 1 || width == 8 || width == 16 || width == 32 || width == 64;
    }
    return elementType.isF16() || elementType.isBF16() || elementType.isF32() ||
           mlir::isa<mlir::Float8E4M3FNType, mlir::Float8E5M2Type, HiFloat8Type, Float8E8M0Type>(elementType) ||
           isFp4PairType(elementType);
}

// The scalar carrier of a tla.full value: Bool/hif8/e8m0/fp4x2 broadcast an i8
// byte, unsigned elements broadcast their same-width signless twin, and every
// other element type broadcasts itself.
static bool isFullScalarCarrier(mlir::Type scalarType, mlir::Type elementType)
{
    if (scalarType == elementType)
        return true;
    if (isBoolElementType(elementType) || mlir::isa<HiFloat8Type, Float8E8M0Type>(elementType) ||
        isFp4PairType(elementType))
        return scalarType.isInteger(8);
    if (auto elemInt = mlir::dyn_cast<mlir::IntegerType>(elementType); elemInt && elemInt.isUnsigned()) {
        auto scalarInt = mlir::dyn_cast<mlir::IntegerType>(scalarType);
        return scalarInt && scalarInt.isSignless() && scalarInt.getWidth() == elemInt.getWidth();
    }
    return scalarType == elementType;
}

// tla.full broadcasts a scalar, or a *one-lane* vector fragment such as a
// tla.reduce result. A full-width VectorSSA source is rejected: the lowering
// would silently splat lane 0 and drop the rest. The optional predicate must
// match the result, checked with the same helper the binary ops use.
mlir::LogicalResult FullOp::verify()
{
    if (!hasEnclosingRegion<VecFuncOp>(getOperation()))
        return emitOpError("must be nested inside a tla.vec.func region");
    if (auto region = getOperation()->getParentOfType<VecFuncOp>())
        if (auto mode = region->getAttrOfType<mlir::StringAttr>("mode"))
            if (!mode.getValue().equals_insensitive("simd"))
                return emitOpError("is only supported in tla.vec.func(mode='simd')");
    auto resultType = mlir::dyn_cast<VectorSSAType>(getResult().getType());
    if (!resultType)
        return emitOpError("expected a !tla.vector result");
    if (!isSupportedFullElementType(resultType.getElementType()))
        return emitOpError() << "unsupported full element type " << resultType.getElementType();
    if (auto srcType = mlir::dyn_cast<VectorSSAType>(getValue().getType())) {
        if (srcType.getElementType() != resultType.getElementType())
            return emitOpError() << "vector source element type " << srcType.getElementType()
                                 << " does not match result element type " << resultType.getElementType();
        auto lanes = srcType.getValidLanes();
        if (lanes != 1)
            return emitOpError() << "vector source must be a one-lane fragment, got " << lanes
                                 << " valid lanes; broadcasting a full-width vector would keep "
                                    "lane 0 only";
    } else if (!isFullScalarCarrier(getValue().getType(), resultType.getElementType())) {
        return emitOpError() << "scalar source type " << getValue().getType() << " is not a valid carrier for "
                             << resultType.getElementType();
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

mlir::LogicalResult ExpSubOp::verify()
{
    if (!hasEnclosingRegion<VecFuncOp>(getOperation()))
        return emitOpError("must be nested inside a tla.vec.func region");
    VectorSSAType lhsType = getLhs().getType();
    VectorSSAType rhsType = getRhs().getType();
    VectorSSAType resultType = getResult().getType();
    MaskSSAType maskType = getMask().getType();
    auto elem = lhsType.getElementType();
    if ((!elem.isF16() && !elem.isF32()) || rhsType.getElementType() != elem)
        return emitOpError("requires matching f16 or f32 input element types");
    int64_t lanes = elem.isF16() ? 128 : 64;
    if (!resultType.getElementType().isF32() || resultType.getValidLanes() != 64)
        return emitOpError("requires a 64-lane f32 result vector");
    if (maskType.getPhysicalLanes() != lanes)
        return emitOpError("requires a ") << lanes << "-lane input-domain mask";
    if (getPart() != 0 && getPart() != 1)
        return emitOpError("part must be 0 or 1");
    return mlir::success();
}

static mlir::LogicalResult verifyShiftSourceAndResult(
    mlir::Operation* op, VectorSSAType sourceType, VectorSSAType resultType, mlir::Value mask)
{
    auto sourceElement = mlir::dyn_cast<mlir::IntegerType>(sourceType.getElementType());
    if (!sourceElement || sourceElement.isSigned() ||
        (sourceElement.getWidth() != 8 && sourceElement.getWidth() != 16 && sourceElement.getWidth() != 32 &&
         sourceElement.getWidth() != 64))
        return op->emitOpError("requires an i8/i16/i32/i64 or ui8/ui16/ui32/ui64 source vector");
    if (sourceElement.getWidth() == 64 && sourceType.getValidLanes() != 32)
        return op->emitOpError("64-bit shifts require a full 256-byte register (32 lanes)");
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
        if (getSource2() || getStoreDist() || getBlockStride())
            return emitOpError("second source, store_dist and block_stride are not supported when storing !tla.mask");
        if (getUnalignedUbAccess())
            return emitOpError("unaligned_ub_access is not supported when storing !tla.mask");
        // AscendC StoreAlign<__ubuf__ T*, MaskReg>: SupportBytes<T,1,2,4,8>;
        // Bool (i1) is b8 storage (elementStorageBytes → 1).
        int64_t destElemBytes = elementStorageBytes(destType.getPtr().getPointee());
        if (destElemBytes != 1 && destElemBytes != 2 && destElemBytes != 4 && destElemBytes != 8)
            return emitOpError(
                "dest !tla.tensor element type must be a b8/b16/b32/b64 scalar "
                "for MaskSSA store");
        return mlir::success();
    }

    auto vectorSource = mlir::dyn_cast<VectorSSAType>(getSource().getType());
    if (!vectorSource)
        return emitOpError("source must be !tla.vector or !tla.mask");

    if ((vectorSource.getElementType().isInteger(64) || vectorSource.getElementType().isF64()) &&
        vectorSource.getValidLanes() != 32)
        return emitOpError("64-bit register store requires a full 256-byte vector register (32 lanes)");

    bool isUnaligned = getUnalignedUbAccess().value_or(false);
    if (isUnaligned && getBlockStride())
        return emitOpError("unaligned_ub_access and block_stride are mutually exclusive");
    if (isUnaligned && getMask())
        return emitOpError("unaligned_ub_access uses the destination extent and does not accept a predicate mask");

    auto destElementType = destType.getPtr().getPointee();
    auto sourceElementType = vectorSource.getElementType();
    if (elementStorageBytes(sourceElementType) == 8 || elementStorageBytes(destElementType) == 8) {
        if (sourceElementType != destElementType)
            return emitOpError("64-bit register store requires matching source and destination element types");
        if (getBlockStride())
            return emitOpError("64-bit register store does not support block_stride");
    }
    if (auto storeDist = getStoreDist()) {
        ::StoreDist dist = storeDist->getStoreDist();
        bool isNormal = dist == ::StoreDist::norm;
        bool isFirstElement = dist == ::StoreDist::first_element_b8 || dist == ::StoreDist::first_element_b16 ||
                              dist == ::StoreDist::first_element_b32;
        bool isIntlv =
            dist == ::StoreDist::intlv_b8 || dist == ::StoreDist::intlv_b16 || dist == ::StoreDist::intlv_b32;
        // AscendC pack_* is same-type DistVST (T == ActualT). Only pack4_b32
        // still remaps numeric widths in the DSL model.
        bool remapsElementWidth = dist == ::StoreDist::pack4_b32;
        bool isPack = dist == ::StoreDist::pack_b16 || dist == ::StoreDist::pack_b32 || dist == ::StoreDist::pack_b64 ||
                      dist == ::StoreDist::pack4_b32;
        if (!isNormal && isUnaligned)
            return emitOpError("non-normal store_dist is incompatible with unaligned_ub_access");
        if (!isNormal && getBlockStride())
            return emitOpError("non-normal store_dist is incompatible with block_stride");
        if (isFirstElement && getMask())
            return emitOpError("predicate mask is not supported with first_element store_dist");
        if (isIntlv && getMask())
            return emitOpError("predicate mask is not supported with intlv store_dist");

        mlir::Value source2 = getSource2();
        if (isIntlv) {
            if (!source2)
                return emitOpError("intlv store_dist requires a second !tla.vector source");
            auto second = mlir::dyn_cast<VectorSSAType>(source2.getType());
            if (!second || second != vectorSource)
                return emitOpError("intlv store_dist requires matching !tla.vector source types");
            if (vectorSource.getElementType() != destElementType)
                return emitOpError("intlv store source and destination element types must match");
            // Bool (i1) is b8 storage — use elementStorageBytes, not getByteSize
            // (which is 0 for i1 and would divide-by-zero).
            int64_t elemBytes = elementStorageBytes(vectorSource.getElementType());
            if (elemBytes <= 0 || 256 % elemBytes != 0)
                return emitOpError("intlv store_dist requires a supported element width");
            int64_t expectedLanes = 256 / elemBytes;
            if (vectorSource.getValidLanes() != expectedLanes)
                return emitOpError("intlv store_dist requires two identical full VL vectors");
            if (dist == ::StoreDist::intlv_b8 && elemBytes != 1)
                return emitOpError("intlv_b8 requires a 1-byte element type");
            if (dist == ::StoreDist::intlv_b16 && elemBytes != 2)
                return emitOpError("intlv_b16 requires a 2-byte element type");
            if (dist == ::StoreDist::intlv_b32 && elemBytes != 4)
                return emitOpError("intlv_b32 requires a 4-byte element type");
        } else if (source2) {
            return emitOpError("second source is only valid with intlv_b8/intlv_b16/intlv_b32 store_dist");
        }

        if (isPack && !remapsElementWidth && vectorSource.getElementType() != destElementType &&
            elementStorageBytes(vectorSource.getElementType()) == elementStorageBytes(destElementType))
            return emitOpError() << stringifyStoreDist(dist)
                                 << " requires matching source and destination element types";

        // Pack / first_element / intlv always need a supported source→dest
        // pairing, including when an explicit mask is present.
        if ((isPack || isFirstElement || isIntlv) &&
            !getPackedStorePredicateGeometry(dist, vectorSource.getElementType(), destElementType)) {
            return emitOpError() << "unsupported " << stringifyStoreDist(dist)
                                 << " source/destination element widths: source "
                                 << getBitSizeOfFixedWidthScalarType(vectorSource.getElementType())
                                 << " bits, destination " << getBitSizeOfFixedWidthScalarType(destElementType)
                                 << " bits; supported layouts are pack_b16 16->16 or 8->8, "
                                    "pack_b32 32->32 or 16->16, pack_b64 64->64, "
                                    "pack4_b32 {8,16,32}->8 or 32->32, first_element_b8 8->8, "
                                    "first_element_b16 16->16, first_element_b32 32->32, and same-width intlv_*";
        }
        // NORM / same-width pack / first_element: only Bool mismatches are
        // rejected here so historical same-width signedness differences
        // (e.g. ui32→i32, i16→u16 histograms) keep working. Frontend matches.
        if (sourceElementType != destElementType &&
            (isBoolElementType(sourceElementType) || isBoolElementType(destElementType)))
            return emitOpError("store source and destination element types must match");
    } else {
        if (getSource2())
            return emitOpError("second source is only valid with intlv_b8/intlv_b16/intlv_b32 store_dist");
        if (vectorSource.getElementType() != destElementType &&
            (isBoolElementType(vectorSource.getElementType()) || isBoolElementType(destElementType))) {
            return emitOpError("store source and destination element types must match");
        }
    }
    return verifyMaskMatchesVector(getOperation(), getMask(), vectorSource);
}

static mlir::LogicalResult verifyMaskProducerType(mlir::Operation* op, MaskSSAType maskType, mlir::Type elementType)
{
    constexpr int64_t kVectorRegisterBytes = 256;
    // Bool (i1) uses elementStorageBytes → 1B/lane (AscendC CreateMask<bool>/pset_b8).
    int64_t elementBytes = elementStorageBytes(elementType);
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

mlir::LogicalResult GenerateMaskOp::verify()
{
    if (!hasEnclosingRegion<VecFuncOp>(getOperation()))
        return emitOpError("must be nested inside a tla.vec.func region");
    if (auto region = getOperation()->getParentOfType<VecFuncOp>())
        if (auto mode = region->getAttrOfType<mlir::StringAttr>("mode"))
            if (!mode.getValue().equals_insensitive("simd"))
                return emitOpError("is only supported in tla.vec.func(mode='simd')");

    auto srcType = mlir::dyn_cast<VectorSSAType>(getSrc().getType());
    if (!srcType)
        return emitOpError("src must be !tla.vector");
    // Only the element bit width matters: the register is a bit array, offset
    // selects a block of 256/sizeof(T) bits, and each bit of that block becomes
    // one predicate lane. Any 16- or 32-bit integer or float element qualifies.
    mlir::Type elementType = srcType.getElementType();
    if (!elementType.isIntOrFloat())
        return emitOpError() << "unsupported source element type " << elementType
                             << "; generate_mask only supports 16- and 32-bit elements";
    int64_t elementBits = elementType.getIntOrFloatBitWidth();
    if (elementBits != 16 && elementBits != 32)
        return emitOpError() << "unsupported source element type " << elementType
                             << "; generate_mask only supports 16- and 32-bit elements";

    if (failed(verifyMaskProducerType(getOperation(), getResult().getType(), elementType)))
        return mlir::failure();

    int64_t maxOffset = elementBits == 16 ? 15 : 31;
    int64_t offset = getOffset();
    if (offset < 0 || offset > maxOffset)
        return emitOpError() << "offset must be in 0.." << maxOffset << " for a b" << elementBits << " source, got "
                             << offset;
    return mlir::success();
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

    if (xIsVector) {
        auto xVec = mlir::cast<VectorSSAType>(getX().getType());
        auto yVec = mlir::dyn_cast<VectorSSAType>(getY().getType());
        auto resultVec = mlir::dyn_cast<VectorSSAType>(getResult().getType());
        if (!isSupportedSelectElementType(xVec.getElementType()))
            return emitOpError() << "unsupported select element type " << xVec.getElementType();
        if (!yVec || !resultVec || yVec.getElementType() != xVec.getElementType() ||
            resultVec.getElementType() != xVec.getElementType())
            return emitOpError("requires x, y, and result to have the same VectorSSA element type");
        return verifyMaskMatchesVector(getOperation(), getMask(), xVec);
    }

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
        if (isBoolElementType(srcElem) || isBoolElementType(resultTy.getElementType()))
            return emitOpError("Bool VectorSSA is move-only (load/store); gather is not supported");
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

    if (isBoolElementType(resultTy.getElementType()) ||
        isBoolElementType(mlir::cast<TlaTensorType>(getX().getType()).getPtr().getPointee()))
        return emitOpError("Bool VectorSSA is move-only (load/store); gather is not supported");

    auto indexElementType = indexType.getElementType();
    auto integerType = mlir::dyn_cast<mlir::IntegerType>(indexElementType);
    if (!integerType || !integerType.isSignless() || integerType.getWidth() != 32)
        return emitOpError() << "gather indices must be i32, got " << indexElementType;
    return verifyMaskMatchesVector(getOperation(), getMask(), getResult().getType());
}

mlir::LogicalResult ScatterOp::verify()
{
    if (!hasEnclosingRegion<VecFuncOp>(getOperation()))
        return emitOpError("must be nested inside a tla.vec.func region");
    if (auto vf = getOperation()->getParentOfType<VecFuncOp>()) {
        if (auto mode = vf->getAttrOfType<mlir::StringAttr>("mode");
            mode && !mode.getValue().equals_insensitive("simd"))
            return emitOpError("requires a SIMD tla.vec.func region");
    }

    auto dstType = getDst().getType();
    if (dstType.getPtr().getAddrspace() != ::AddressSpace::ub)
        return emitOpError("destination must reside in UB");
    auto srcType = getSrc().getType();
    mlir::Type element = srcType.getElementType();
    if (dstType.getPtr().getPointee() != element)
        return emitOpError("source and destination element types must match");
    if (isBoolElementType(element) || isBoolElementType(dstType.getPtr().getPointee()))
        return emitOpError("Bool VectorSSA is move-only (load/store); scatter is not supported");
    unsigned bits = 0;
    if (auto integer = mlir::dyn_cast<mlir::IntegerType>(element)) {
        bits = integer.getWidth();
        if (bits != 8 && bits != 16 && bits != 32 && bits != 64)
            return emitOpError("unsupported data element type: ") << element;
    } else if (element.isF16() || element.isBF16()) {
        bits = 16;
    } else if (element.isF32()) {
        bits = 32;
    } else {
        return emitOpError("unsupported data element type: ") << element;
    }
    auto indexType = getIndices().getType();
    auto indexElement = mlir::dyn_cast<mlir::IntegerType>(indexType.getElementType());
    unsigned indexBits = bits <= 16 ? 16 : 32;
    if (!indexElement || !indexElement.isUnsigned() ||
        (indexElement.getWidth() != indexBits && !(bits == 64 && indexElement.getWidth() == 64)))
        return emitOpError("indices must be unsigned u16 for b8/b16, u32 for b32, or u32/u64 for b64 data");
    int64_t sourceLanes = srcType.getValidLanes();
    int64_t indexLanes = indexType.getValidLanes();
    if (sourceLanes != mlir::ShapedType::kDynamic && indexLanes != mlir::ShapedType::kDynamic) {
        int64_t required = bits == 8 ? (sourceLanes + 1) / 2 : sourceLanes;
        if (indexLanes < required)
            return emitOpError("indices have fewer valid lanes than the source requires");
    }
    if (getMask()) {
        int64_t expected = 2048 / std::max(bits, 16u);
        if (getMask().getType().getPhysicalLanes() != expected)
            return emitOpError("mask has incorrect predicate lane count; expected ") << expected;
    }
    return mlir::success();
}

namespace {
struct SqueezeStateResource : mlir::SideEffects::Resource::Base<SqueezeStateResource> {
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(SqueezeStateResource)
    llvm::StringRef getName() final
    {
        return "tla.squeeze_state";
    }
};
} // namespace

void SqueezeOp::getEffects(llvm::SmallVectorImpl<mlir::MemoryEffects::EffectInstance>& effects)
{
    if (getStoreReg()) {
        effects.emplace_back(mlir::MemoryEffects::Read::get(), SqueezeStateResource::get());
        effects.emplace_back(mlir::MemoryEffects::Write::get(), SqueezeStateResource::get());
    }
}

mlir::Speculation::Speculatability SqueezeOp::getSpeculatability()
{
    return getStoreReg() ? mlir::Speculation::NotSpeculatable : mlir::Speculation::Speculatable;
}

void ClearSqueezeStateOp::getEffects(llvm::SmallVectorImpl<mlir::MemoryEffects::EffectInstance>& effects)
{
    effects.emplace_back(mlir::MemoryEffects::Write::get(), SqueezeStateResource::get());
}

void StoreUnalignUpdateOp::getEffects(llvm::SmallVectorImpl<mlir::MemoryEffects::EffectInstance>& effects)
{
    if (getCompressed()) {
        effects.emplace_back(mlir::MemoryEffects::Read::get(), SqueezeStateResource::get());
        effects.emplace_back(mlir::MemoryEffects::Write::get(), SqueezeStateResource::get());
    }
    effects.emplace_back(
        mlir::MemoryEffects::Write::get(), &getOperation()->getOpOperand(0), mlir::SideEffects::DefaultResource::get());
}

void StoreUnalignPostOp::getEffects(llvm::SmallVectorImpl<mlir::MemoryEffects::EffectInstance>& effects)
{
    if (getCompressed()) {
        effects.emplace_back(mlir::MemoryEffects::Read::get(), SqueezeStateResource::get());
        effects.emplace_back(mlir::MemoryEffects::Write::get(), SqueezeStateResource::get());
    }
    if (getDest().getType().isInteger(64))
        effects.emplace_back(mlir::MemoryEffects::Write::get(), mlir::SideEffects::DefaultResource::get());
    else
        effects.emplace_back(
            mlir::MemoryEffects::Write::get(), &getOperation()->getOpOperand(1),
            mlir::SideEffects::DefaultResource::get());
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
    if (mlir::isa<::tla::HiFloat8Type>(source) || mlir::isa<::tla::HiFloat8Type>(dest)) {
        if (trait.size() != 3 || trait[0] < 0 || trait[0] > 3 || trait[1] < 0 || trait[1] > 2 || trait[2] < 0 ||
            !::symbolizeRoundMode(trait[2]))
            return emitOpError("trait must contain valid register-slot, saturation and rounding codes");
        if (!((source.isF32() && mlir::isa<HiFloat8Type>(dest)) || (mlir::isa<HiFloat8Type>(source) && dest.isF32())))
            return emitOpError("HiFloat8 pairs with Float32; compose other routes explicitly");
        if (mlir::isa<HiFloat8Type>(dest) && trait[2] != static_cast<int32_t>(::RoundMode::cast_round) &&
            trait[2] != static_cast<int32_t>(::RoundMode::cast_hybrid))
            return emitOpError("FP32->HiF8 supports CAST_ROUND or CAST_HYBRID");
    }
    if (isFp4PairType(source) || isFp4PairType(dest)) {
        if (trait.size() != 3 || trait[0] < 0 || trait[0] > 3 || trait[1] < 0 || trait[1] > 2 || trait[2] < 0 ||
            !::symbolizeRoundMode(trait[2]))
            return emitOpError("trait must contain valid register-slot, saturation and rounding codes");
        bool narrow = source.isBF16() && isFp4PairType(dest);
        if (!narrow && !(isFp4PairType(source) && dest.isBF16()))
            return emitOpError("FP4x2 pairs with BFloat16; compose other routes explicitly");
        if (narrow && trait[1] != 0 && trait[1] != 1)
            return emitOpError("BF16->FP4x2 requires SAT or UNKNOWN");
        if (narrow && trait[2] == static_cast<int32_t>(::RoundMode::cast_odd))
            return emitOpError(
                "BF16->FP4x2 supports CAST_ROUND/CAST_FLOOR/CAST_CEIL/CAST_TRUNC/CAST_RINT/CAST_NONE only");
        if (narrow) {
            if (getMask() && getMask().getType().getPhysicalLanes() != 64 &&
                getMask().getType().getPhysicalLanes() != 128)
                return emitOpError("BF16->FP4x2 requires a 128-lane BF16 source mask or a 64-lane pair mask");
            return mlir::success();
        }
    }
    return verifyMaskMatchesVector(getOperation(), getMask(), getSource().getType());
}

mlir::LogicalResult BitCastOp::verify()
{
    if (!hasEnclosingRegion<VecFuncOp>(getOperation()))
        return emitOpError("must be nested inside a tla.vec.func region");

    VectorSSAType sourceType = getSource().getType();
    VectorSSAType resultType = getResult().getType();
    // Bool keeps an i1 IR identity but is move-only; do not treat i1 as a 1-bit
    // register lane for bitcast width math (AscendC storage is b8 / 1 byte).
    if (isBoolElementType(sourceType.getElementType()) || isBoolElementType(resultType.getElementType()))
        return emitOpError("Bool VectorSSA is move-only (load/store); bitcast is not supported");
    int64_t sourceBits = getBitSizeOfFixedWidthScalarType(sourceType.getElementType());
    int64_t resultBits = getBitSizeOfFixedWidthScalarType(resultType.getElementType());
    if (sourceBits <= 0 || resultBits <= 0)
        return emitOpError() << "unsupported element type: source " << sourceType.getElementType() << ", result "
                             << resultType.getElementType();
    int64_t sourceLanes = sourceType.getValidLanes();
    int64_t resultLanes = resultType.getValidLanes();
    bool sourceDynamic = sourceLanes == mlir::ShapedType::kDynamic;
    bool resultDynamic = resultLanes == mlir::ShapedType::kDynamic;
    if (sourceDynamic != resultDynamic || (!sourceDynamic && sourceLanes * sourceBits != resultLanes * resultBits))
        return emitOpError("preserves the valid bit count; both lane counts must be dynamic or describe equal bits");

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

mlir::LogicalResult PairWiseOp::verify()
{
    if (!hasEnclosingRegion<VecFuncOp>(getOperation()))
        return emitOpError("must be nested inside a tla.vec.func region");
    auto parent = getOperation()->getParentOfType<VecFuncOp>();
    if (auto mode = parent->getAttrOfType<mlir::StringAttr>("mode");
        mode && !mode.getValue().equals_insensitive("simd"))
        return emitOpError("pair_wise is only supported in SIMD mode");

    if (getKind() != "add")
        return emitOpError() << "kind must be \"add\" for pair_wise operation, got \"" << getKind() << "\"";

    auto sourceType = mlir::dyn_cast<VectorSSAType>(getOperand().getType());
    if (!sourceType)
        return emitOpError("operand must be !tla.vector");
    auto resultType = mlir::dyn_cast<VectorSSAType>(getResult().getType());
    if (!resultType)
        return emitOpError("result must be !tla.vector");
    if (resultType != sourceType)
        return emitOpError("result type must match operand type");

    mlir::Type elem = sourceType.getElementType();
    if (!mlir::isa<mlir::Float16Type, mlir::Float32Type>(elem))
        return emitOpError() << "unsupported pair_wise element type " << elem << " (supported: f16/f32)";

    return verifyMaskMatchesVector(getOperation(), getMask(), sourceType);
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
        auto dist = getLoadDist() ? getLoadDist()->getLoadDist() : ::LoadDist::norm;
        if (dist != ::LoadDist::norm && dist != ::LoadDist::us && dist != ::LoadDist::ds)
            return emitOpError("mask load only supports load_dist norm|us|ds (AscendC MaskDist NORM/US/DS)");
        if (getBlockStride() || getRepeatStride())
            return emitOpError("block_stride/repeat_stride is not supported when loading !tla.mask");
        if (getUnalignedUbAccess())
            return emitOpError("unaligned_ub_access is not supported when loading !tla.mask");
        int64_t sourceElemBytes = elementStorageBytes(sourceType.getPtr().getPointee());
        if (sourceElemBytes != 1 && sourceElemBytes != 2 && sourceElemBytes != 4 && sourceElemBytes != 8)
            return emitOpError("source !tla.tensor element type must be a b8/b16/b32/b64 scalar for MaskSSA load");
        int64_t lanes = maskResult.getPhysicalLanes();
        if (lanes != 32 && lanes != 64 && lanes != 128 && lanes != 256)
            return emitOpError() << "unsupported !tla.mask lane count " << lanes;
        return mlir::success();
    }

    if (!mlir::isa<VectorSSAType>(getResult().getType()))
        return emitOpError("result must be !tla.vector or !tla.mask");

    auto vectorResult = mlir::cast<VectorSSAType>(getResult().getType());
    if (vectorResult.getElementType().isF64() && vectorResult.getValidLanes() != 32)
        return emitOpError("f64 register load requires a full 256-byte vector register (32 lanes)");
    if (sourceType.getPtr().getPointee().isInteger(64) || sourceType.getPtr().getPointee().isF64() ||
        vectorResult.getElementType().isInteger(64) || vectorResult.getElementType().isF64()) {
        if (vectorResult.getElementType() != sourceType.getPtr().getPointee())
            return emitOpError("64-bit register load requires matching element types");
        if ((getLoadDist() && getLoadDist()->getLoadDist() != LoadDist::norm) || getBlockStride())
            return emitOpError("64-bit register load requires NORM distribution");
        if (getUnalignedUbAccess().value_or(false) && vectorResult.getValidLanes() != 32)
            return emitOpError("64-bit unaligned load requires a full 256-byte vector register (32 lanes)");
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

    auto dist = getLoadDist() ? getLoadDist()->getLoadDist() : ::LoadDist::norm;
    const bool dual = dist == ::LoadDist::dintlv_b8 || dist == ::LoadDist::dintlv_b16 || dist == ::LoadDist::dintlv_b32;
    if (dual && !getResult2())
        return emitOpError() << "load_dist " << stringifyLoadDist(dist) << " requires a second !tla.vector result";
    if (!dual && getResult2())
        return emitOpError("second result is only valid as !tla.vector with a dintlv_b8/b16/b32 load_dist");
    if (dist == ::LoadDist::dintlv_b8 && isInt4PairType(sourceType.getPtr().getPointee()))
        return emitOpError("dintlv_b8 does not support Int4x2 register pairs");

    int64_t width;
    switch (dist) {
        case ::LoadDist::norm:
            return mlir::success();
        case ::LoadDist::brc_b8:
        case ::LoadDist::us_b8:
        case ::LoadDist::ds_b8:
        case ::LoadDist::unpack_b8:
        case ::LoadDist::unpack4_b8:
        case ::LoadDist::dintlv_b8:
            width = 1;
            break;
        case ::LoadDist::brc_b16:
        case ::LoadDist::us_b16:
        case ::LoadDist::ds_b16:
        case ::LoadDist::unpack_b16:
        case ::LoadDist::e2b_b16:
        case ::LoadDist::dintlv_b16:
            width = 2;
            break;
        case ::LoadDist::brc_b32:
        case ::LoadDist::unpack_b32:
        case ::LoadDist::e2b_b32:
        case ::LoadDist::dintlv_b32:
            width = 4;
            break;
        case ::LoadDist::blk:
            width = elementStorageBytes(sourceType.getPtr().getPointee());
            if (width != 1 && width != 2 && width != 4)
                return emitOpError("blk requires a 1/2/4-byte source element type");
            break;
        default:
            return emitOpError("load_dist must be a vector distribution");
    }
    if (getUnalignedUbAccess().value_or(false))
        return emitOpError() << "load_dist " << stringifyLoadDist(dist) << " requires aligned UB access";
    auto sourceElem = sourceType.getPtr().getPointee();
    const int64_t sourceBytes = elementStorageBytes(sourceElem);
    if (sourceBytes != width)
        return emitOpError() << stringifyLoadDist(dist) << " requires a " << width << "-byte source element type";
    const int64_t lanes = 256 / width;
    if (dual && (getResult2().getType() != vectorResult || vectorResult.getElementType() != sourceElem ||
                 vectorResult.getValidLanes() != lanes))
        return emitOpError() << stringifyLoadDist(dist) << " requires two identical full " << lanes
                             << "-lane vectors matching the source type";
    if (!dual && (vectorResult.getElementType() != sourceElem || vectorResult.getValidLanes() != lanes))
        return emitOpError() << stringifyLoadDist(dist) << " requires a full " << lanes
                             << "-lane vector matching the source type";
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

/// UnAlign UPDATE stride advances up to one VL per register in the operation.
static mlir::LogicalResult verifyUnalignUpdateStride(
    mlir::Operation* op, mlir::Value stride, mlir::Type elemType, bool dual)
{
    if (failed(verifyUpdateStride(op, stride)))
        return mlir::failure();
    auto constantOp = stride.getDefiningOp<mlir::arith::ConstantOp>();
    if (!constantOp)
        return mlir::success();
    auto intAttr = mlir::dyn_cast<mlir::IntegerAttr>(constantOp.getValue());
    if (!intAttr)
        return mlir::success();
    // SoT: Bool i1 is b8 (1), not getByteSize(i1)==0.
    int64_t elemBytes = elementStorageBytes(elemType);
    if (elemBytes <= 0)
        return mlir::success();
    constexpr int64_t kVectorRegisterBytes = 256;
    int64_t vl = kVectorRegisterBytes / elemBytes;
    if (intAttr.getInt() > (dual ? 2 * vl : vl))
        return op->emitOpError("post_update_stride must be in (0, ")
               << (dual ? "2 * VL" : "VL") << "] for the dtype (VL=" << vl << " elements), got " << intAttr.getInt();
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

static constexpr int64_t kDualUpdateAlignment = 32;

static mlir::Value foldIntegerExpression(mlir::Value value, mlir::OpBuilder& scratch, mlir::IRMapping& constants)
{
    if (mlir::Value cached = constants.lookupOrNull(value))
        return cached;
    mlir::IntegerAttr attr;
    if (mlir::matchPattern(value, mlir::m_Constant(&attr))) {
        mlir::Value constant = scratch.create<mlir::arith::ConstantOp>(value.getLoc(), attr);
        constants.map(value, constant);
        return constant;
    }

    mlir::Operation* producer = value.getDefiningOp();
    if (!producer || producer->getName().getDialectNamespace() != mlir::arith::ArithDialect::getDialectNamespace() ||
        !value.getType().isIntOrIndex() || producer->getNumResults() != 1)
        return {};
    for (mlir::Value operand : producer->getOperands())
        if (!foldIntegerExpression(operand, scratch, constants))
            return {};

    // Fold an isolated copy because a fold hook may update its operation in place.
    mlir::Operation* copy = scratch.clone(*producer, constants);
    llvm::SmallVector<mlir::Value, 1> folded;
    if (mlir::failed(scratch.tryFold(copy, folded)) || folded.size() != 1 ||
        !mlir::matchPattern(folded.front(), mlir::m_Constant(&attr)))
        return {};
    constants.map(value, folded.front());
    return folded.front();
}

static std::optional<int64_t> getStaticInteger(mlir::Value value)
{
    mlir::IntegerAttr attr;
    if (!mlir::matchPattern(value, mlir::m_Constant(&attr))) {
        mlir::Block scratch;
        mlir::OpBuilder builder(value.getContext());
        builder.setInsertionPointToEnd(&scratch);
        mlir::IRMapping constants;
        if (mlir::Value folded = foldIntegerExpression(value, builder, constants))
            mlir::matchPattern(folded, mlir::m_Constant(&attr));
    }
    if (attr && attr.getValue().isSignedIntN(64))
        return attr.getInt();
    return std::nullopt;
}

static std::optional<int64_t> getDualUpdateTensorResidue(mlir::Value tensor, mlir::Value relativeTo = {});

static std::optional<int64_t> getDualUpdatePointerResidue(mlir::Value pointer, mlir::Value relativeTo)
{
    auto pointerType = mlir::dyn_cast<PtrType>(pointer.getType());
    if (!pointerType)
        return std::nullopt;
    if (auto add = pointer.getDefiningOp<PtrAddOp>()) {
        auto base = getDualUpdatePointerResidue(add.getPtr(), relativeTo);
        auto offset = getStaticInteger(add.getOffset());
        auto elemBytes = elementStorageBytes(pointerType.getPointee());
        if (base && offset && elemBytes > 0)
            return (*base + (*offset % kDualUpdateAlignment) * elemBytes) % kDualUpdateAlignment;
        return std::nullopt;
    }
    if (auto recast = pointer.getDefiningOp<RecastPtrOp>())
        return getDualUpdatePointerResidue(recast.getSrc(), relativeTo);
    if (auto tensorPtr = pointer.getDefiningOp<TensorPtrOp>()) {
        auto backing = resolveTensorBacking(tensorPtr.getSrc());
        if (succeeded(backing))
            return getDualUpdatePointerResidue(*backing, relativeTo);
        return std::nullopt;
    }
    if (relativeTo) {
        mlir::Value window, stride;
        if (auto load = pointer.getDefiningOp<LoadUpdateOp>(); load && !load.getBlockStride()) {
            window = load.getSource();
            stride = load.getPostUpdateStride();
        } else if (auto store = pointer.getDefiningOp<StoreUpdateOp>(); store && !store.getBlockStride()) {
            window = store.getDest();
            stride = store.getPostUpdateStride();
        } else {
            return std::nullopt;
        }
        auto base = getDualUpdateTensorResidue(window, relativeTo);
        auto advance = getStaticInteger(stride);
        auto elemBytes = elementStorageBytes(pointerType.getPointee());
        if (base && advance && elemBytes > 0)
            return (*base + (*advance % kDualUpdateAlignment) * elemBytes) % kDualUpdateAlignment;
        return std::nullopt;
    }
    if (auto fromInt = pointer.getDefiningOp<IntToPtrOp>()) {
        auto address = getStaticInteger(fromInt.getOperand());
        if (address)
            return *address % kDualUpdateAlignment;
        return std::nullopt;
    }
    // UPDATE results carry the same pointer type even if the advance changes
    // alignment. An arbitrary block argument can carry such a result too.
    if (pointer.getDefiningOp<AllocPtrOp>() && pointerType.getAlignment() % kDualUpdateAlignment == 0)
        return 0;
    return std::nullopt;
}

static std::optional<int64_t> getDualUpdateTensorResidue(mlir::Value tensor, mlir::Value relativeTo)
{
    if (tensor == relativeTo)
        return 0;
    auto type = mlir::cast<TlaTensorType>(tensor.getType());
    if (!isLinearLayout(type.getLayout().getLayoutTag()))
        return std::nullopt;
    if (auto argument = mlir::dyn_cast<mlir::BlockArgument>(tensor)) {
        // A loop argument inherits its initial residue only when the backedge
        // is derived from that argument and advances it by a multiple of 32 bytes.
        auto loop = mlir::dyn_cast<mlir::scf::ForOp>(argument.getOwner()->getParentOp());
        if (relativeTo || !loop || argument.getArgNumber() == 0)
            return std::nullopt;
        unsigned index = argument.getArgNumber() - 1;
        auto yield = mlir::cast<mlir::scf::YieldOp>(loop.getBody()->getTerminator());
        auto delta = getDualUpdateTensorResidue(yield.getOperand(index), argument);
        if (!delta || *delta != 0)
            return std::nullopt;
        return getDualUpdateTensorResidue(loop.getInitArgs()[index]);
    }
    mlir::Value pointer;
    llvm::SmallVector<int64_t, 2> coord, stride;
    if (auto desc = tensor.getDefiningOp<TensorDescOp>()) {
        pointer = desc.getBase();
        coord = {
            getStaticInteger(desc.getCoord0()).value_or(mlir::ShapedType::kDynamic),
            getStaticInteger(desc.getCoord1()).value_or(mlir::ShapedType::kDynamic)};
        stride = {
            getStaticInteger(desc.getStride0()).value_or(mlir::ShapedType::kDynamic),
            getStaticInteger(desc.getStride1()).value_or(mlir::ShapedType::kDynamic)};
    } else {
        if (auto make = tensor.getDefiningOp<MakeTensorOp>())
            pointer = make.getPtr();
        else if (auto make = tensor.getDefiningOp<MakeTensorLikeOp>())
            pointer = make.getPtr();
        else
            return std::nullopt;
        if (failed(getTlaIndexTreeLeaves(type.getCoord().getTree(), coord)) ||
            failed(getTlaIndexTreeLeaves(type.getLayout().getStride().getTree(), stride)))
            return std::nullopt;
    }
    auto base = getDualUpdatePointerResidue(pointer, relativeTo);
    if (!base || coord.size() != stride.size())
        return std::nullopt;
    int64_t elemOffset = 0;
    for (auto [c, s] : llvm::zip_equal(coord, stride)) {
        if (c == 0 || s == 0)
            continue;
        if (mlir::ShapedType::isDynamic(c) || mlir::ShapedType::isDynamic(s))
            return std::nullopt;
        elemOffset += (c % kDualUpdateAlignment) * (s % kDualUpdateAlignment);
    }
    auto elemBytes = elementStorageBytes(type.getPtr().getPointee());
    return (*base + elemOffset * elemBytes) % kDualUpdateAlignment;
}

static mlir::LogicalResult verifyDualUpdate(
    mlir::Operation* op, mlir::Value window, int64_t distBytes, mlir::Type elem, VectorSSAType first,
    mlir::Value second, mlir::Value stride)
{
    int64_t elemBytes = elementStorageBytes(elem);
    bool usesB32Carrier = distBytes == 4 && (elem.isInteger(64) || elem.isF64());
    if (elemBytes != distBytes && !usesB32Carrier)
        return op->emitOpError("dual UPDATE distribution does not match the element width");
    if (!second || second.getType() != first || first.getValidLanes() != 256 / elemBytes)
        return op->emitOpError("dual UPDATE requires two identical full VL vectors");
    if (auto address = getDualUpdateTensorResidue(window); address && *address != 0)
        return op->emitOpError("dual UPDATE effective UB address must be 32-byte aligned");
    if (auto advance = getStaticInteger(stride); advance && *advance % (kDualUpdateAlignment / elemBytes) != 0)
        return op->emitOpError("dual UPDATE post_update_stride in bytes must be 32-byte aligned");
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
    auto resultVec = mlir::dyn_cast<VectorSSAType>(getResult().getType());
    if (!resultVec)
        return emitOpError("result must be !tla.vector");
    mlir::Type srcElem = sourceType.getPtr().getPointee();
    if (resultVec.getElementType() != srcElem)
        return emitOpError("load_update source and result element types must match");
    if (auto dist = getLoadDist()) {
        if (getBlockStride())
            return emitOpError("load_dist is mutually exclusive with block_stride");
        int64_t distBytes;
        switch (dist->getLoadDist()) {
            case ::LoadDist::dintlv_b8:
                distBytes = 1;
                break;
            case ::LoadDist::dintlv_b16:
                distBytes = 2;
                break;
            case ::LoadDist::dintlv_b32:
                distBytes = 4;
                break;
            default:
                return emitOpError("load_update load_dist must be a DINTLV distribution");
        }
        if (failed(verifyDualUpdate(
                getOperation(), getSource(), distBytes, srcElem, resultVec, getResult2(), getPostUpdateStride())))
            return mlir::failure();
    } else if (getResult2()) {
        return emitOpError("second vector result requires a DINTLV load_dist");
    }
    if (failed(verifyOptionalBlockStride(getOperation(), getBlockStride())))
        return mlir::failure();
    if (getBlockStride()) {
        if (failed(verifyDataBlockUpdateStride(getOperation(), getPostUpdateStride())))
            return mlir::failure();
    } else if (failed(verifyUpdateStride(getOperation(), getPostUpdateStride()))) {
        return mlir::failure();
    }
    return verifyUpdatePtr(getOperation(), getNextPtr(), srcElem);
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
    auto sourceVec = mlir::dyn_cast<VectorSSAType>(getSource().getType());
    if (!sourceVec)
        return emitOpError("source must be !tla.vector");
    mlir::Type destElem = destType.getPtr().getPointee();
    if (sourceVec.getElementType() != destElem)
        return emitOpError("store_update source and destination element types must match");
    if (auto dist = getStoreDist()) {
        if (getBlockStride())
            return emitOpError("store_dist is mutually exclusive with block_stride");
        if (getMask())
            return emitOpError("predicate mask is not supported with intlv store_dist");
        int64_t distBytes;
        switch (dist->getStoreDist()) {
            case ::StoreDist::intlv_b8:
                distBytes = 1;
                break;
            case ::StoreDist::intlv_b16:
                distBytes = 2;
                break;
            case ::StoreDist::intlv_b32:
                distBytes = 4;
                break;
            default:
                return emitOpError("store_update store_dist must be an INTLV distribution");
        }
        if (failed(verifyDualUpdate(
                getOperation(), getDest(), distBytes, destElem, sourceVec, getSource2(), getPostUpdateStride())))
            return mlir::failure();
    } else if (getSource2()) {
        return emitOpError("second vector source requires an INTLV store_dist");
    }
    if (failed(verifyOptionalBlockStride(getOperation(), getBlockStride())))
        return mlir::failure();
    if (getBlockStride()) {
        if (failed(verifyDataBlockUpdateStride(getOperation(), getPostUpdateStride())))
            return mlir::failure();
    } else if (failed(verifyUpdateStride(getOperation(), getPostUpdateStride()))) {
        return mlir::failure();
    }
    if (failed(verifyUpdatePtr(getOperation(), getNextPtr(), destElem)))
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

static mlir::LogicalResult verifyUnalignDualVectors(mlir::Operation* op, VectorSSAType first, mlir::Value second)
{
    auto elemBytes = elementStorageBytes(first.getElementType());
    if (elemBytes != 1 && elemBytes != 2 && elemBytes != 4 && elemBytes != 8)
        return op->emitOpError("dual unaligned UPDATE requires b8/b16/b32/b64 storage");
    if (second.getType() != first || first.getValidLanes() != 256 / elemBytes)
        return op->emitOpError("dual unaligned UPDATE requires two identical full VL vectors");
    return mlir::success();
}

mlir::LogicalResult LoadUnalignUpdateOp::verify()
{
    if (failed(verifyUnalignUbTensor(getOperation(), getSource(), "source")))
        return mlir::failure();
    auto resultVec = mlir::dyn_cast<VectorSSAType>(getResult().getType());
    if (!resultVec)
        return emitOpError("result must be !tla.vector");
    auto sourceType = mlir::cast<TlaTensorType>(getSource().getType());
    mlir::Type elemType = sourceType.getPtr().getPointee();
    if (resultVec.getElementType() != elemType)
        return emitOpError("load_unalign_update source and result element types must match");
    // Bool UnAlign Load UPDATE: AscendC casts to int8_t (SupportBytes b8).
    if (getResult2() && failed(verifyUnalignDualVectors(getOperation(), resultVec, getResult2())))
        return mlir::failure();
    if (failed(verifyUnalignUpdateStride(getOperation(), getPostUpdateStride(), elemType, bool(getResult2()))))
        return mlir::failure();
    return verifyUpdatePtr(getOperation(), getNextPtr(), elemType);
}

// UPDATE uses an origin window. Recover known byte offsets from its backing
// pointer, including ptr_add, whose type alone may describe the allocation.
static std::optional<uint64_t> getStaticMaskStoreAddressRemainder(mlir::Value value, uint64_t alignment)
{
    uint64_t byteOffset = 0;
    while (value) {
        if (auto view = value.getDefiningOp<TileViewOp>()) {
            value = view.getSource();
        } else if (auto make = value.getDefiningOp<MakeTensorOp>()) {
            value = make.getPtr();
        } else if (auto like = value.getDefiningOp<MakeTensorLikeOp>()) {
            value = like.getPtr();
        } else if (auto desc = value.getDefiningOp<TensorDescOp>();
                   desc && mlir::isa<PtrType>(desc.getBase().getType())) {
            value = desc.getBase();
        } else if (auto tensorPtr = value.getDefiningOp<TensorPtrOp>()) {
            value = tensorPtr.getSrc();
        } else if (auto recast = value.getDefiningOp<RecastPtrOp>()) {
            value = recast.getSrc();
        } else if (auto add = value.getDefiningOp<PtrAddOp>()) {
            llvm::APInt offset;
            if (!mlir::matchPattern(add.getOffset(), mlir::m_ConstantInt(&offset)))
                return std::nullopt;
            int64_t elemBytes = elementStorageBytes(add.getPtr().getType().getPointee());
            if (elemBytes <= 0)
                return std::nullopt;
            byteOffset += (offset.sextOrTrunc(64).getZExtValue() % alignment) * elemBytes;
            value = add.getPtr();
        } else if (auto ptrFromInt = value.getDefiningOp<IntToPtrOp>()) {
            llvm::APInt address;
            if (!mlir::matchPattern(ptrFromInt.getAddr(), mlir::m_ConstantInt(&address)))
                return std::nullopt;
            return (address.sextOrTrunc(64).getZExtValue() + byteOffset) % alignment;
        } else if (auto ifOp = value.getDefiningOp<mlir::scf::IfOp>()) {
            // Pointer types can retain the allocation alignment after offsets.
            // A branch result only proves the remainder common to both yields.
            unsigned index = mlir::cast<mlir::OpResult>(value).getResultNumber();
            std::optional<uint64_t> remainder;
            for (mlir::Region& region : ifOp->getRegions()) {
                if (!region.hasOneBlock() || !region.front().mightHaveTerminator())
                    return std::nullopt;
                auto yield = mlir::dyn_cast<mlir::scf::YieldOp>(region.front().getTerminator());
                if (!yield || index >= yield.getNumOperands())
                    return std::nullopt;
                auto branchRemainder = getStaticMaskStoreAddressRemainder(yield.getOperand(index), alignment);
                if (!branchRemainder || (remainder && remainder != branchRemainder))
                    return std::nullopt;
                remainder = branchRemainder;
            }
            return (*remainder + byteOffset) % alignment;
        } else {
            // A store fed by a loop-carried window can use the seed on the
            // first iteration, including when the store is in a nested branch.
            // Inspect the seed before relying on the carried type.
            if (auto arg = mlir::dyn_cast<mlir::BlockArgument>(value); arg && arg.getArgNumber() > 0) {
                if (auto loop = mlir::dyn_cast<mlir::scf::ForOp>(arg.getOwner()->getParentOp());
                    loop && arg.getArgNumber() <= loop.getInitArgs().size()) {
                    value = loop.getInitArgs()[arg.getArgNumber() - 1];
                    continue;
                }
            }
            auto ptr = mlir::dyn_cast<PtrType>(value.getType());
            if (auto tensor = mlir::dyn_cast<TlaTensorType>(value.getType()))
                ptr = tensor.getPtr();
            if (ptr && ptr.getAlignment() != 0 && ptr.getAlignment() % alignment == 0)
                return byteOffset % alignment;
            return std::nullopt;
        }
    }
    return std::nullopt;
}

mlir::LogicalResult StoreUnalignUpdateOp::verify()
{
    if (failed(verifyUnalignUbTensor(getOperation(), getDest(), "dest")))
        return mlir::failure();
    auto destType = mlir::cast<TlaTensorType>(getDest().getType());
    mlir::Type elemType = destType.getPtr().getPointee();
    if (auto sourceMask = mlir::dyn_cast<MaskSSAType>(getSource().getType())) {
        if (getSource2())
            return emitOpError("unaligned mask store requires a single mask source");
        if (getCompressed())
            return emitOpError("compressed UPDATE requires a vector source");
        auto region = getOperation()->getParentOfType<VecFuncOp>();
        if (auto mode = region->getAttrOfType<mlir::StringAttr>("mode");
            mode && !mode.getValue().equals_insensitive("simd"))
            return emitOpError("unaligned mask store requires a SIMD tla.vec.func region");
        int64_t elemBytes = elementStorageBytes(elemType);
        if (elemBytes != 2 && elemBytes != 4)
            return emitOpError("unaligned mask store destination must have b16 or b32 elements");
        constexpr int64_t kVectorRegisterBytes = 256;
        int64_t lanes = kVectorRegisterBytes / elemBytes;
        if (sourceMask.getPhysicalLanes() != lanes)
            return emitOpError("unaligned mask store requires a mask with ") << lanes << " lanes";
        int64_t packedBytes = lanes / 8;
        llvm::APInt stride;
        if (!mlir::matchPattern(getPostUpdateStride(), mlir::m_ConstantInt(&stride)) ||
            stride != packedBytes / elemBytes)
            return emitOpError("unaligned mask store requires a constant post_update_stride of ")
                   << packedBytes / elemBytes << " elements";
        if (auto remainder = getStaticMaskStoreAddressRemainder(getDest(), packedBytes); remainder && *remainder != 0)
            return emitOpError("unaligned mask store requires a ") << packedBytes << "-byte-aligned UB address";
        return verifyUpdatePtr(getOperation(), getNextPtr(), elemType);
    }
    auto sourceVec = mlir::cast<VectorSSAType>(getSource().getType());
    if (sourceVec.getElementType() != elemType)
        return emitOpError("store_unalign_update source and destination element types must match");
    if (getCompressed()) {
        if (getSource2())
            return emitOpError("compressed UPDATE requires a single vector");
        if (isBoolElementType(elemType))
            return emitOpError("compressed UPDATE does not support Bool elements");
    } else {
        if (getSource2() && failed(verifyUnalignDualVectors(getOperation(), sourceVec, getSource2())))
            return mlir::failure();
        if (failed(verifyUnalignUpdateStride(getOperation(), getPostUpdateStride(), elemType, bool(getSource2()))))
            return mlir::failure();
    }
    return verifyUpdatePtr(getOperation(), getNextPtr(), elemType);
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
mlir::LogicalResult GetSystemCycleOp::verify()
{
    if (hasEnclosing<VecFuncOp>(getOperation()))
        return emitOpError("must execute on Scalar outside tla.vec.func");
    if (!hasEnclosingRegion<CubeOp>(getOperation()) && !hasEnclosingRegion<VectorOp>(getOperation()))
        return emitOpError("must be nested inside a tla.cube or tla.vector region");
    return mlir::success();
}

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

mlir::LogicalResult DataSyncBarrierOp::verify()
{
    if ((*this)->getParentOfType<VecFuncOp>())
        return emitOpError("must not be nested inside a tla.vec.func region; move the barrier outside the region");
    // Lexical check only: the op is re-verified after tla-lower-func moves it
    // into func.func, where the tla.cube wrapper is legitimately gone and
    // hasEnclosingRegion would report every lowered func as a cube.
    if (getDomain() == DsbTarget::ub && hasEnclosing<CubeOp>(getOperation()))
        return emitOpError(
            "ub domain is not issued on AIC; use gm, gm_ub or seq inside tla.cube, or move the barrier into "
            "tla.vector");
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
    llvm::SmallVector<int64_t, 4> srcOrigin, dstOrigin;
    if (mlir::failed(::tla::getTlaIndexTreeLeaves(srcTy.getLayout().getOrigin().getTree(), srcOrigin)) ||
        mlir::failed(::tla::getTlaIndexTreeLeaves(dstTy.getLayout().getOrigin().getTree(), dstOrigin)))
        return mlir::failure();
    if (srcOrigin.size() == 3 || dstOrigin.size() == 3) {
        if (srcOrigin.size() != 3 || dstOrigin.size() != 3 || src != AddressSpace::gm || dst != AddressSpace::l1)
            return emitOpError("batch-matrix copy currently requires two rank-3 GM/L1 tensors");
        auto elem = srcTy.getPtr().getPointee();
        if ((!elem.isFloat8E4M3FN() && !elem.isFloat8E5M2() && !mlir::isa<Float8E8M0Type>(elem)) ||
            elem != dstTy.getPtr().getPointee())
            return emitOpError("batch-matrix copy supports matching FP8 or E8M0 element types");
        if (!isSupportedBatchCopyLayout(srcTy.getLayout().getLayoutTag(), dstTy.getLayout().getLayoutTag(), elem))
            return emitOpError("unsupported batch-matrix copy layout conversion");
        if (srcOrigin[0] != mlir::ShapedType::kDynamic && (srcOrigin[0] < 0 || srcOrigin[0] > 65535))
            return emitOpError("batch count must fit the unsigned 16-bit hardware field");
        llvm::SmallVector<int64_t, 5> srcStride, dstStride;
        if (mlir::failed(getTlaIndexTreeLeaves(srcTy.getLayout().getStride().getTree(), srcStride)) ||
            mlir::failed(getTlaIndexTreeLeaves(dstTy.getLayout().getStride().getTree(), dstStride)))
            return mlir::failure();
        if (dstStride[0] != mlir::ShapedType::kDynamic &&
            (dstStride[0] <= 0 || dstStride[0] % 32 || dstStride[0] / 32 > 65535))
            return emitOpError("destination batch stride must fit the unsigned 16-bit 32-byte-block field");
        if (mlir::isa<Float8E8M0Type>(elem) && srcStride[0] != mlir::ShapedType::kDynamic && srcStride[0] % 2)
            return emitOpError("E8M0 source batch stride must preserve exponent pairs");
        for (size_t i = 0; i < 3; ++i)
            if (srcOrigin[i] != mlir::ShapedType::kDynamic && dstOrigin[i] != mlir::ShapedType::kDynamic &&
                srcOrigin[i] != dstOrigin[i])
                return emitOpError("batch-matrix copy logical extents must match");
    }

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
        if (dst != AddressSpace::gm)
            return emitOpError("atomic operation requires dst to be in GM address space");

        if (src != AddressSpace::ub && src != AddressSpace::l0c)
            return emitOpError("atomic writeback requires a UB or L0C source");

        auto elemType = dstTy.getPtr().getPointee();
        if (!elemType.isF32() && !elemType.isF16() && !elemType.isBF16() && !elemType.isSignlessInteger(32) &&
            !elemType.isSignlessInteger(16) && !elemType.isSignlessInteger(8))
            return emitOpError(
                "atomic operation requires dst element type to be one of "
                "f32, f16, bf16, i32, i16, i8");
    }

    // L2 cache hints only apply to the GM endpoint of a copy.
    if (getSrcCachePolicyAttr() && src != AddressSpace::gm)
        return emitOpError("src_cache_policy requires src to be in GM address space");
    if (getDstCachePolicyAttr() && dst != AddressSpace::gm)
        return emitOpError("dst_cache_policy requires dst to be in GM address space");

    // Optional params is either L0C fixpipe knobs or CopyPadParams (GM→UB / GM→L1 VL).
    if (mlir::Value params = getParams()) {
        mlir::Type paramsTy = params.getType();
        if (mlir::isa<CopyPadParamsType>(paramsTy)) {
            const bool gmToUb = src == AddressSpace::gm && dst == AddressSpace::ub;
            const bool gmToL1 = src == AddressSpace::gm && dst == AddressSpace::l1;
            if (!(gmToUb || gmToL1))
                return emitOpError("copy_pad_params require GM→UB or GM→L1 copy route");
            if (gmToL1) {
                // Pad ABI is only registered for VectorLayout→VectorLayout; other
                // GM→L1 layouts would silently drop pad trailing operands.
                if (srcTy.getLayout().getLayoutTag() != LayoutTag::VectorLayout ||
                    dstTy.getLayout().getLayoutTag() != LayoutTag::VectorLayout)
                    return emitOpError("copy_pad_params on GM→L1 require VectorLayout→VectorLayout");
                // fp8/fp4 VL stubs exist for exact copy, but padEnable=1 falls through to
                // the two-arg path in Cube BC — reject here to match frontend whitelist.
                mlir::Type padElem = srcTy.getPtr().getPointee();
                if (padElem.isFloat8E4M3FN() || padElem.isFloat8E5M2() || mlir::isa<::tla::Float8E8M0Type>(padElem) ||
                    ::tla::isPackedFp4Type(padElem))
                    return emitOpError("copy_pad_params on GM→L1 does not support fp8/fp4 element types");
            }
            if (!params.getDefiningOp<CopyPadParamsOp>())
                return emitOpError("copy_pad_params must be produced by tla.CopyPadParams");
        } else if (mlir::isa<CopyL0C2DstParamsType>(paramsTy)) {
            if (src != AddressSpace::l0c)
                return emitOpError("copy_l0c2dst_params require an L0C source");
        }
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
        if (!isCompareScalarTypeCompatible(rhsType, lhsElementType))
            return emitOpError() << "scalar operand must have element type " << lhsElementType
                                 << " or its same-width signless carrier, got " << rhsType;
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

mlir::LogicalResult ArchShuffleOp::verify()
{
    // The shuffle reads another lane's register, which only exists in a SIMT
    // region; outside one there is no warp and nothing lowers the op, so an
    // unchecked shuffle would survive the pipeline unlowered.
    auto region = getOperation()->getParentOfType<VecFuncOp>();
    auto regionMode = region ? region->getAttrOfType<mlir::StringAttr>("mode") : mlir::StringAttr{};
    if (!regionMode || !regionMode.getValue().equals_insensitive("simt"))
        return emitOpError("must be nested inside a tla.vec.func with mode='simt'");

    const int64_t width = getWidth();
    // A group must tile the 32-lane warp exactly, so only powers of two divide it
    // evenly. width == 1 would make every shuffle a no-op read of the lane itself.
    if (width < 2 || width > 32 || (width & (width - 1)) != 0)
        return emitOpError() << "width must be a power of two in [2, 32], got " << width;

    // Only a constant index can be checked here; a runtime index is the caller's
    // responsibility, and the hardware masks it to 5 bits either way.
    mlir::APInt indexValue;
    if (!mlir::matchPattern(getIndex(), mlir::m_ConstantInt(&indexValue)))
        return mlir::success();
    const int64_t index = indexValue.getSExtValue();
    if (index < 0)
        return emitOpError() << "index must be non-negative, got " << index;
    // The index travels in a 5-bit control field, so 32 and above is where it
    // wraps: 33 becomes 1 and reads a neighbour. Below 32 nothing is lost, and a
    // distance or mask that reaches outside the lane's group is answered by the
    // hardware keeping the lane's own value -- a legal request, not an error.
    // "idx" is exempt because there the index is a lane number and the masking
    // is exactly the documented src_lane % width.
    if (getMode() != ::ShuffleMode::idx && index >= kWarpSize)
        return emitOpError() << "index must be less than " << kWarpSize << " for this mode, got " << index;
    return mlir::success();
}

} // namespace tla
