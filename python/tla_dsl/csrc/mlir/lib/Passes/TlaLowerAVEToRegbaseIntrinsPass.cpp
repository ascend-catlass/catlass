#include "PassesCommon.h"
#include "PassesInternal.h"

#include "bishengir/Dialect/HIVMAVE/IR/HIVMAVE.h"

namespace tla {
namespace {

static Value castToLaneMask(OpBuilder& builder, Location loc, Value mask)
{
    auto maskType = dyn_cast<VectorType>(mask.getType());
    if (maskType && maskType.getShape().back() == 256)
        return mask;
    auto predType = VectorType::get({256}, builder.getI1Type());
    return builder.create<UnrealizedConversionCastOp>(loc, predType, mask)->getResult(0);
}

static Value castFromLaneMask(OpBuilder& builder, Location loc, Type resultType, Value mask)
{
    if (mask.getType() == resultType)
        return mask;
    return builder.create<UnrealizedConversionCastOp>(loc, resultType, mask)->getResult(0);
}

template <typename SignedOp, typename UnsignedOp>
static Value createScalarCmpIntrinsic(OpBuilder& builder, hivmave::VFCmpS op, Type resultType, bool isUnsigned)
{
    Location loc = op.getLoc();
    Value mask = castToLaneMask(builder, loc, op.getMask());
    auto predType = VectorType::get({256}, builder.getI1Type());
    Operation* intrinsic = isUnsigned ? builder.create<UnsignedOp>(loc, predType, op.getVec(), op.getScalar(), mask) :
                                        builder.create<SignedOp>(loc, predType, op.getVec(), op.getScalar(), mask);
    Value result = intrinsic->getResult(0);
    return castFromLaneMask(builder, loc, resultType, result);
}

class TlaLowerAVEToRegbaseIntrinsPass : public PassWrapper<TlaLowerAVEToRegbaseIntrinsPass, OperationPass<ModuleOp>> {
public:
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TlaLowerAVEToRegbaseIntrinsPass)

    StringRef getArgument() const override
    {
        return "tla-lower-ave-to-regbase-intrins";
    }
    StringRef getName() const override
    {
        return "TlaLowerAVEToRegbaseIntrinsPass";
    }

    void runOnOperation() override
    {
        IRRewriter rewriter(&getContext());
        bool failed = false;
        getOperation().walk([&](hivmave::VFCmpS op) {
            if (failed)
                return;
            auto vecType = dyn_cast<VectorType>(op.getVec().getType());
            if (!vecType) {
                op.emitError("expected vector operand");
                failed = true;
                return;
            }
            Type elementType = vecType.getElementType();
            bool isUnsigned = elementType.isUnsignedInteger();
            Value result;
            rewriter.setInsertionPoint(op);
            switch (op.getCmp()) {
                case hivmave::CmpType::EQ:
                    result = createScalarCmpIntrinsic<
                        hivm_regbaseintrins::VCmpsSZEqInstrOp, hivm_regbaseintrins::VCmpsUZEqInstrOp>(
                        rewriter, op, op.getRes().getType(), isUnsigned);
                    break;
                case hivmave::CmpType::NE:
                    result = createScalarCmpIntrinsic<
                        hivm_regbaseintrins::VCmpsSZNeInstrOp, hivm_regbaseintrins::VCmpsUZNeInstrOp>(
                        rewriter, op, op.getRes().getType(), isUnsigned);
                    break;
                case hivmave::CmpType::GT:
                    result = createScalarCmpIntrinsic<
                        hivm_regbaseintrins::VCmpsSZGtInstrOp, hivm_regbaseintrins::VCmpsUZGtInstrOp>(
                        rewriter, op, op.getRes().getType(), isUnsigned);
                    break;
                case hivmave::CmpType::GE:
                    result = createScalarCmpIntrinsic<
                        hivm_regbaseintrins::VCmpsSZGeInstrOp, hivm_regbaseintrins::VCmpsUZGeInstrOp>(
                        rewriter, op, op.getRes().getType(), isUnsigned);
                    break;
                case hivmave::CmpType::LT:
                    result = createScalarCmpIntrinsic<
                        hivm_regbaseintrins::VCmpsSZLtInstrOp, hivm_regbaseintrins::VCmpsUZLtInstrOp>(
                        rewriter, op, op.getRes().getType(), isUnsigned);
                    break;
                case hivmave::CmpType::LE:
                    result = createScalarCmpIntrinsic<
                        hivm_regbaseintrins::VCmpsSZLeInstrOp, hivm_regbaseintrins::VCmpsUZLeInstrOp>(
                        rewriter, op, op.getRes().getType(), isUnsigned);
                    break;
                // Unsigned relational modes are already selected by the TLA
                // pass (VCmpU*); the scalar twin must follow the same rule.
                case hivmave::CmpType::ULT:
                    result = createScalarCmpIntrinsic<
                        hivm_regbaseintrins::VCmpsUZLtInstrOp, hivm_regbaseintrins::VCmpsUZLtInstrOp>(
                        rewriter, op, op.getRes().getType(), /*isUnsigned=*/true);
                    break;
                case hivmave::CmpType::ULE:
                    result = createScalarCmpIntrinsic<
                        hivm_regbaseintrins::VCmpsUZLeInstrOp, hivm_regbaseintrins::VCmpsUZLeInstrOp>(
                        rewriter, op, op.getRes().getType(), /*isUnsigned=*/true);
                    break;
                case hivmave::CmpType::UGT:
                    result = createScalarCmpIntrinsic<
                        hivm_regbaseintrins::VCmpsUZGtInstrOp, hivm_regbaseintrins::VCmpsUZGtInstrOp>(
                        rewriter, op, op.getRes().getType(), /*isUnsigned=*/true);
                    break;
                case hivmave::CmpType::UGE:
                    result = createScalarCmpIntrinsic<
                        hivm_regbaseintrins::VCmpsUZGeInstrOp, hivm_regbaseintrins::VCmpsUZGeInstrOp>(
                        rewriter, op, op.getRes().getType(), /*isUnsigned=*/true);
                    break;
                default:
                    op.emitError("unsupported scalar compare mode");
                    failed = true;
                    return;
            }
            rewriter.replaceOp(op, result);
        });
        if (failed)
            signalPassFailure();
    }
};

} // namespace

std::unique_ptr<Pass> createTlaLowerAVEToRegbaseIntrinsPass()
{
    return std::make_unique<TlaLowerAVEToRegbaseIntrinsPass>();
}

void registerTlaLowerAVEToRegbaseIntrinsPass()
{
    PassRegistration<TlaLowerAVEToRegbaseIntrinsPass>();
}

} // namespace tla
