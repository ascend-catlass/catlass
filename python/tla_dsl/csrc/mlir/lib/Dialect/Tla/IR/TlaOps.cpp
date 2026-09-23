#include "Dialect/Tla/IR/TlaOps.h"

#include <array>
#include <limits>
#include <numeric>
#include <optional>

#include "Dialect/Tla/IR/TlaAttrs.h"
#include "Dialect/Tla/IR/TlaTypes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
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
    return intType && (intType.isSignless() || intType.isUnsigned()) && intType.getWidth() == 32;
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
    bool isLinear = layoutTag == LayoutTag::RowMajor || layoutTag == LayoutTag::ColumnMajor;
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
// tla.func to func.func -- and tla-vector-region / TlaSplitMixedFuncPass
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
    bool inVector = hasEnclosing<VectorOp>(getOperation());
    bool inCube = hasEnclosing<CubeOp>(getOperation());
    if (inVector == inCube)
        return emitOpError("must be nested inside exactly one tla.vector or tla.cube region");
    if (inVector && hasEnclosing<VecFuncOp>(getOperation()))
        return emitOpError("must be outside tla.vec.func");
    for (mlir::Type type : getOperandTypes()) {
        if (mlir::isa<PtrType, mlir::IntegerType, mlir::FloatType>(type))
            continue;
        return emitOpError("operands must be !tla.ptr or scalar integer/float (not index), got ") << type;
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
    // Fill exists for the MX pad use case: zeroing a packed fp4/fp8 operand
    // tile. Wider types and the e8m0 scale format have no fill route in the
    // bc layer, so refuse them here rather than at lowering time.
    ::mlir::Type elem = dstTy.getPtr().getPointee();
    if (!::tla::isPackedFp4Type(elem) && !mlir::isa<::mlir::Float8E4M3FNType, ::mlir::Float8E5M2Type>(elem))
        return emitOpError("fill destination element type must be fp4 or fp8");
    ::llvm::APInt valueBits;
    if (mlir::matchPattern(getValue(), mlir::m_ConstantInt(&valueBits)) && !valueBits.isZero())
        return emitOpError("fill value must be 0");
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

        // If `storeDistAttr` supports intlv mode, check here if `getResult2()` is None
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
        if ((!getMask() || isFirstElement) &&
            !getPackedStorePredicateGeometry(dist, vectorSource.getElementType(), destElementType)) {
            return emitOpError() << "unsupported " << stringifyStoreDist(dist)
                                 << " source/destination element widths: source "
                                 << getBitSizeOfFixedWidthScalarType(vectorSource.getElementType())
                                 << " bits, destination " << getBitSizeOfFixedWidthScalarType(destElementType)
                                 << " bits; supported layouts are pack_b32 32->16, pack_b32 16->16, pack_b16 16->8, "
                                    "first_element_b8 8->8, first_element_b16 16->16, and first_element_b32 32->32";
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

mlir::LogicalResult WhereOp::verify()
{
    if (!hasEnclosingRegion<VecFuncOp>(getOperation()))
        return emitOpError("must be nested inside a tla.vec.func region");
    return verifyMaskMatchesVector(getOperation(), getMask(), getX().getType());
}

mlir::LogicalResult GatherOp::verify()
{
    if (!hasEnclosingRegion<VecFuncOp>(getOperation()))
        return emitOpError("must be nested inside a tla.vec.func region");
    auto indexType = mlir::dyn_cast<VectorSSAType>(getY().getType());
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

mlir::LogicalResult CastOp::verify()
{
    if (!hasEnclosingRegion<VecFuncOp>(getOperation()))
        return emitOpError("must be nested inside a tla.vec.func region");
    return verifyMaskMatchesVector(getOperation(), getMask(), getSource().getType());
}

mlir::LogicalResult ReduceOp::verify()
{
    if (!hasEnclosingRegion<VecFuncOp>(getOperation()))
        return emitOpError("must be nested inside a tla.vec.func region");
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
        if (getLoadDist())
            return emitOpError("load_dist is not supported when loading !tla.mask");
        if (getBlockStride() || getRepeatStride())
            return emitOpError("block_stride/repeat_stride is not supported when loading !tla.mask");
        if (getUnalignedUbAccess())
            return emitOpError("unaligned_ub_access is not supported when loading !tla.mask");
        int64_t sourceElemBytes = getByteSizeOfFixedWidthScalarType(sourceType.getPtr().getPointee());
        if (sourceElemBytes != 1 && sourceElemBytes != 2 && sourceElemBytes != 4)
            return emitOpError(
                "source !tla.tensor element type must be a 1/2/4-byte scalar "
                "for MaskSSA load");
        int64_t lanes = maskResult.getPhysicalLanes();
        if (lanes != 32 && lanes != 64 && lanes != 128 && lanes != 256)
            return emitOpError() << "unsupported !tla.mask lane count " << lanes;
        return mlir::success();
    }

    if (!mlir::isa<VectorSSAType>(getResult().getType()))
        return emitOpError("result must be !tla.vector or !tla.mask");

    // Strided block load (vsldb): mutually exclusive with load_dist, the dual
    // result, and the MaskSSA path (checked above: mask result rejects
    // load_dist but block_stride is rejected here for clarity).
    if (auto blockStrideAttr = getBlockStrideAttr()) {
        if (getLoadDist())
            return emitOpError("block_stride is mutually exclusive with load_dist");
        if (getUnalignedUbAccess())
            return emitOpError("block_stride is mutually exclusive with unaligned_ub_access");
        if (getResult2())
            return emitOpError("block_stride is not valid with a second (dual-destination) result");
        // vsldb packs each stride into a 16-bit immediate field.
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

    if (isDintlv) {
        if (!getResult2())
            return emitOpError("load_dist dintlv_b32 requires a second result (dual-destination load)");
    } else if (getResult2()) {
        return emitOpError("second result is only valid with load_dist dintlv_b32");
    }
    return mlir::success();
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

template <typename OpTy>
static mlir::LogicalResult verifyCrossCoreFlagOp(OpTy op)
{
    if (!hasEnclosingRegion<CubeOp>(op.getOperation()) && !hasEnclosingRegion<VectorOp>(op.getOperation()))
        return op.emitOpError("must be nested inside a tla.cube or tla.vector region");
    auto flagType = mlir::cast<CrossFlagType>(op.getFlag().getType());
    auto aivIdAttr = op->template getAttrOfType<mlir::IntegerAttr>("aiv_id");
    if (flagType.getMode() == 4) {
        if (!aivIdAttr || (aivIdAttr.getInt() != 0 && aivIdAttr.getInt() != 1))
            return op.emitOpError("mode 4 requires aiv_id to be the compile-time integer 0 or 1");
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
    bool vectorRoute = (src == AddressSpace::gm && dst == AddressSpace::ub) ||
                       (src == AddressSpace::ub && dst == AddressSpace::gm) ||
                       (src == AddressSpace::ub && dst == AddressSpace::l1);

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

mlir::LogicalResult PrintTensorOp::verify()
{
    // CANN's 1 MiB debug FIFO reserves 48 bytes for the shape TLV and 72 bytes
    // for the tensor TLV. Its 32-byte payload alignment leaves 262112 f32 values:
    // floor((1 MiB - 48 - 72) / 32) * (32 / sizeof(float)).
    constexpr int64_t kMaxFloat32Elements = 262112;
    auto tensorType = getValue().getType();
    auto ptr = tensorType.getPtr();
    auto elementType = ptr.getPointee();
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
    if (layoutTag != LayoutTag::RowMajor && layoutTag != LayoutTag::ColumnMajor)
        return op->emitOpError() << tensorRole << " layout must be RowMajor or ColumnMajor";

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
