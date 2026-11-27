#include "Passes/TlaVectorRegionCommon.h"
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
#include "mlir/IR/Matchers.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include <algorithm>
#include <limits>

namespace tla {
namespace vecregion {

LogicalResult lowerLocalMemBar(OpBuilder& b, ::tla::LocalMemBarOp op)
{
    int64_t barrierKind = op.getBarrierKind();
    if (barrierKind < 0 || barrierKind > 11)
        return op.emitError("barrier_kind ") << barrierKind << " is out of range [0, 11]";
    Value encoded = b.create<arith::ConstantIntOp>(op.getLoc(), barrierKind, 32);
    b.create<hivmave::VFMemBarOp>(op.getLoc(), encoded);
    return success();
}

StringRef getVecFuncMode(::tla::VecFuncOp vecFuncOp)
{
    auto modeAttr = vecFuncOp->getAttrOfType<StringAttr>("mode");
    return modeAttr ? modeAttr.getValue() : StringRef("simd");
}

bool isSimtVecFunc(::tla::VecFuncOp vecFuncOp)
{
    return getVecFuncMode(vecFuncOp).equals_insensitive("simt");
}

bool isSimdVecFunc(::tla::VecFuncOp vecFuncOp)
{
    return !isSimtVecFunc(vecFuncOp);
}

void inlineVectorRegionWrappers(func::FuncOp funcOp)
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

bool isIllegalVecFuncArchOp(Operation* op, StringRef& dslName)
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

LogicalResult checkNoArchOpsInVecFunc(func::FuncOp funcOp)
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

namespace {

class LowerCopyPattern : public OpRewritePattern<::tla::CopyOp> {
public:
    LowerCopyPattern(MLIRContext* context, DenseMap<Value, Value>& loweredMemrefByValue)
        : OpRewritePattern<::tla::CopyOp>(context, /*benefit=*/3), loweredMemrefByValue(loweredMemrefByValue)
    {}

    LogicalResult matchAndRewrite(::tla::CopyOp copyOp, PatternRewriter& rewriter) const override
    {
        // Operands: dst, src, optional params (L0C knobs or CopyPadParams).
        if (copyOp->getNumResults() != 0)
            return rewriter.notifyMatchFailure(copyOp, "expected tla.copy with 0 results");
        if (!copyOp.getDst() || !copyOp.getSrc())
            return rewriter.notifyMatchFailure(copyOp, "expected tla.copy dst and src operands");

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

        if (copyOp.getCompact()) {
            auto constant = [](Value value) -> std::optional<int64_t> {
                llvm::APInt bits;
                if (matchPattern(value, m_ConstantInt(&bits)))
                    return bits.getSExtValue();
                return std::nullopt;
            };
            auto equal = [&](Value lhs, Value rhs) {
                // The layout builder may cast the same dynamic extent
                // separately for the shape and stride.
                auto lhsCast = lhs.getDefiningOp<arith::IndexCastOp>();
                auto rhsCast = rhs.getDefiningOp<arith::IndexCastOp>();
                if (lhsCast && rhsCast && lhs.getType() == rhs.getType() && lhsCast.getIn() == rhsCast.getIn())
                    return true;
                auto lhsConstant = constant(lhs);
                return lhs == rhs || (lhsConstant && lhsConstant == constant(rhs));
            };
            if (constant(srcDesc.stride[1]) != 1 || constant(dstDesc.stride[1]) != 1)
                return copyOp.emitOpError("compact requires src and dst column strides of 1");
            if (!equal(srcDesc.originShape[0], dstDesc.originShape[0]) ||
                !equal(srcDesc.originShape[1], dstDesc.originShape[1]))
                return copyOp.emitOpError("compact requires matching valid src and dst extents");
            bool ubToGm = srcDesc.addrspace == "ub";
            const TensorDescriptor& ubDesc = ubToGm ? srcDesc : dstDesc;
            const TensorDescriptor& gmDesc = ubToGm ? dstDesc : srcDesc;
            if (!equal(ubDesc.stride[0], ubDesc.originShape[1]))
                return copyOp.emitOpError("compact requires the UB row stride to equal the valid row width");
            auto columns = constant(gmDesc.originShape[1]);
            auto gmStride = constant(gmDesc.stride[0]);
            if (columns && gmStride && *gmStride < *columns)
                return copyOp.emitOpError("compact requires the GM row stride to cover the valid row width");
            if (::tla::isPackedFp4Type(srcDesc.elementType)) {
                auto isEven = [](std::optional<int64_t> value) { return value && *value % 2 == 0; };
                if (!isEven(columns) || !isEven(gmStride) || !isEven(constant(srcDesc.coord[1])) ||
                    !isEven(constant(dstDesc.coord[1])))
                    return copyOp.emitOpError(
                        "compact FP4 copy requires statically even row width, row strides and column coordinates");
            }
            auto rows = constant(srcDesc.originShape[0]);
            if (rows && (*rows < 0 || *rows > std::numeric_limits<uint16_t>::max()))
                return copyOp.emitOpError("compact row count exceeds the DMA block count range");
            calleeName += "_compact";
        }

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
        operandTypes.reserve(2 + payload.size() + 1);
        for (Value v : payload)
            operandTypes.push_back(v.getType());
        SmallVector<Value, 22> operands = {*srcRuntimeMemref, *dstRuntimeMemref};
        operands.append(payload.begin(), payload.end());
        ::tla::appendGmCachePolicyOperand(
            rewriter, copyOp.getLoc(), copyOp, srcDesc.addrspace, dstDesc.addrspace, operandTypes, operands);
        if (!copyOp.getCompact())
            ::tla::appendGmToUbPadOperands(
                rewriter, copyOp.getLoc(), copyOp, srcDesc.addrspace, dstDesc.addrspace, operandTypes, operands);
        auto callee = ::tla::getOrCreateRuntimeCall(copyOp->getParentOfType<ModuleOp>(), calleeName, operandTypes);

        emitCopyRuntimeCall(rewriter, copyOp, callee, operands, *dstRuntimeMemref);
        rewriter.eraseOp(copyOp);
        return success();
    }

private:
    DenseMap<Value, Value>& loweredMemrefByValue;
};

} // namespace

void populateTlaCopyPatterns(RewritePatternSet& patterns, DenseMap<Value, Value>& loweredMemrefByValue)
{
    patterns.add<LowerCopyPattern>(patterns.getContext(), loweredMemrefByValue);
}

} // namespace vecregion
} // namespace tla
