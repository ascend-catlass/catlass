#include "PassesCommon.h"
#include "PassesInternal.h"

#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/Transforms/FuncConversions.h"
#include "mlir/Dialect/SCF/Transforms/Patterns.h"

namespace tla {
namespace {

static func::FuncOp getOrCreateRuntimeCall(
    ModuleOp module, StringRef name, ArrayRef<Type> operandTypes, Operation* callSite, ArrayRef<Type> resultTypes = {})
{
    auto setCoreTypeFromCaller = [&](func::FuncOp callee) {
        auto caller = callSite ? callSite->getParentOfType<func::FuncOp>() : func::FuncOp();
        if (!caller)
            return;
        auto callerCore = caller->getAttrOfType<hivm::TFuncCoreTypeAttr>(hivm::TFuncCoreTypeAttr::name);
        if (!callerCore)
            return;

        auto calleeCore = callee->getAttrOfType<hivm::TFuncCoreTypeAttr>(hivm::TFuncCoreTypeAttr::name);
        if (!calleeCore) {
            callee->setAttr(hivm::TFuncCoreTypeAttr::name, callerCore);
            return;
        }
        if (calleeCore.getFuncCoreType() != callerCore.getFuncCoreType()) {
            callee->setAttr(
                hivm::TFuncCoreTypeAttr::name,
                hivm::TFuncCoreTypeAttr::get(module.getContext(), hivm::TFuncCoreType::AIC_OR_AIV));
        }
    };

    if (auto existing = module.lookupSymbol<func::FuncOp>(name)) {
        setCoreTypeFromCaller(existing);
        return existing;
    }

    OpBuilder builder(module.getBodyRegion());
    builder.setInsertionPointToStart(module.getBody());
    auto funcType = builder.getFunctionType(operandTypes, resultTypes);
    auto func = builder.create<func::FuncOp>(module.getLoc(), name, funcType);
    func.setPrivate();
    setCoreTypeFromCaller(func);
    return func;
}

static FailureOr<std::string> getMutexPipeSuffix(PipeAttr pipeAttr)
{
    switch (pipeAttr.getPipe()) {
        case Pipe::scalar:
            return std::string("s");
        case Pipe::vector:
            return std::string("v");
        case Pipe::cube:
            return std::string("m");
        case Pipe::mte1:
            return std::string("mte1");
        case Pipe::mte2:
            return std::string("mte2");
        case Pipe::mte3:
            return std::string("mte3");
        case Pipe::fix:
            return std::string("fix");
        default:
            return failure();
    }
}

// get_buf / rls_buf take a different pipe set on each core: ccec accepts
// [S, M, MTE1, MTE2, MTE3, FIX] for --cce-aicore-arch=dav-c310-cube and
// [S, V, MTE2, MTE3] for dav-c310-vec, and the bc stubs mirror that split.
// PIPE_S is the one pipe both cores share, so it is defined in both mutex.cpp
// files, the same way mte2 and mte3 already are. Without this check a
// cube-side #tla.pipe<vector> still lowers to a @get_buf_v call, and the
// mistake surfaces only as an unresolved symbol when the AIC meta-op bitcode
// is linked.
static bool isMutexPipeValidOnCore(Pipe pipe, hivm::TFuncCoreType coreType)
{
    switch (pipe) {
        case Pipe::scalar:
        case Pipe::mte2:
        case Pipe::mte3:
            return true;
        case Pipe::vector:
            return coreType == hivm::TFuncCoreType::AIV;
        case Pipe::cube:
        case Pipe::mte1:
        case Pipe::fix:
            return coreType == hivm::TFuncCoreType::AIC;
        default:
            return false;
    }
}

struct LowerMutexPattern : public OpConversionPattern<::tla::MutexOp> {
    using OpConversionPattern::OpConversionPattern;

    LogicalResult matchAndRewrite(::tla::MutexOp op, OpAdaptor, ConversionPatternRewriter& rewriter) const override
    {
        int64_t mutexId = op.getIdAttr().getInt();
        if (mutexId < 0) {
            op.emitError() << "mutex id auto allocation is not implemented for "
                              "bitcode call lowering";
            return failure();
        }
        if (mutexId > 255) {
            op.emitError() << "mutex id must be in range 0..255 for bitcode call lowering";
            return failure();
        }

        rewriter.replaceOpWithNewOp<arith::ConstantIntOp>(op, mutexId, 8);
        return success();
    }
};

template <typename MutexOpT>
struct LowerMutexAccessPattern : public OpConversionPattern<MutexOpT> {
    using Base = OpConversionPattern<MutexOpT>;
    using OpAdaptor = typename Base::OpAdaptor;

    LowerMutexAccessPattern(TypeConverter& converter, MLIRContext* context, ModuleOp module, StringRef calleePrefix)
        : Base(converter, context), module(module), calleePrefix(calleePrefix.str())
    {}

    LogicalResult matchAndRewrite(MutexOpT op, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override
    {
        Value mutexId = adaptor.getMutex();
        if (!mutexId || !mutexId.getType().isInteger(8))
            return rewriter.notifyMatchFailure(op, "converted mutex id is not i8");

        FailureOr<std::string> pipeSuffix = getMutexPipeSuffix(op.getPipe());
        if (failed(pipeSuffix)) {
            op.emitError() << "unsupported pipe for mutex bitcode call lowering";
            return failure();
        }

        // Only an AIC or AIV classification pins the call to one core. Anything
        // else -- no attribute, or the AIC_OR_AIV stamped on a callee shared by
        // both sides -- names no single core, so there is nothing to check.
        Operation* operation = op.getOperation();
        std::optional<hivm::TFuncCoreType> coreType = getFunctionCoreType(operation->getParentOfType<func::FuncOp>());
        if (coreType && (*coreType == hivm::TFuncCoreType::AIC || *coreType == hivm::TFuncCoreType::AIV) &&
            !isMutexPipeValidOnCore(op.getPipe().getPipe(), *coreType)) {
            bool isAic = *coreType == hivm::TFuncCoreType::AIC;
            op.emitError() << "pipe '" << ::stringifyPipe(op.getPipe().getPipe()) << "' has no mutex form on "
                           << (isAic ? "AIC" : "AIV") << "; valid pipes there are "
                           << (isAic ? "scalar, cube, mte1, mte2, mte3, fix" : "scalar, vector, mte2, mte3");
            return failure();
        }

        SmallVector<Type, 1> operandTypes = {rewriter.getI8Type()};
        auto callee = getOrCreateRuntimeCall(module, calleePrefix + "_" + *pipeSuffix, operandTypes, op.getOperation());
        callee->setAttr("hacc.always_inline", rewriter.getUnitAttr());
        rewriter.create<func::CallOp>(op.getLoc(), callee, ValueRange{mutexId});
        rewriter.eraseOp(op);
        return success();
    }

private:
    ModuleOp module;
    std::string calleePrefix;
};

class TlaLowerMutexToStdPass : public PassWrapper<TlaLowerMutexToStdPass, OperationPass<ModuleOp>> {
public:
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TlaLowerMutexToStdPass)

    StringRef getArgument() const override
    {
        return "tla-lower-mutex-to-std";
    }
    StringRef getName() const override
    {
        return "TlaLowerMutexToStdPass";
    }
    StringRef getDescription() const override
    {
        return "Lower first-class !tla.mutex SSA values to i8 runtime IDs, "
               "including structural SCF/function type conversion.";
    }
    void getDependentDialects(DialectRegistry& registry) const override
    {
        registry.insert<arith::ArithDialect, cf::ControlFlowDialect, func::FuncDialect, scf::SCFDialect>();
    }

    void runOnOperation() override
    {
        ModuleOp module = getOperation();
        MLIRContext* context = &getContext();

        TypeConverter converter;
        converter.addConversion([](Type type) { return type; });
        converter.addConversion([&](::tla::MutexType) -> Type { return IntegerType::get(context, 8); });

        RewritePatternSet patterns(context);
        ConversionTarget target(*context);
        patterns.add<LowerMutexPattern>(converter, context);
        patterns.add<LowerMutexAccessPattern<::tla::MutexLockOp>>(converter, context, module, "get_buf");
        patterns.add<LowerMutexAccessPattern<::tla::MutexUnlockOp>>(converter, context, module, "rls_buf");
        scf::populateSCFStructuralTypeConversionsAndLegality(converter, patterns, target);
        populateFunctionOpInterfaceTypeConversionPattern<func::FuncOp>(patterns, converter);
        populateCallOpTypeConversionPattern(patterns, converter);
        populateBranchOpInterfaceTypeConversionPattern(patterns, converter);
        populateReturnOpTypeConversionPattern(patterns, converter);

        target.addLegalOp<ModuleOp>();
        target.addLegalDialect<arith::ArithDialect, ::tla::TlaDialect>();
        target.addIllegalOp<::tla::MutexOp, ::tla::MutexLockOp, ::tla::MutexUnlockOp>();
        target.addDynamicallyLegalOp<func::FuncOp>([&](func::FuncOp op) {
            return converter.isSignatureLegal(op.getFunctionType()) && converter.isLegal(&op.getBody());
        });
        target.addDynamicallyLegalOp<func::CallOp>(
            [&](func::CallOp op) { return converter.isSignatureLegal(op.getCalleeType()); });
        target.markUnknownOpDynamicallyLegal([&](Operation* op) {
            return isNotBranchOpInterfaceOrReturnLikeOp(op) ||
                   isLegalForBranchOpInterfaceTypeConversionPattern(op, converter) ||
                   isLegalForReturnOpTypeConversionPattern(op, converter);
        });

        if (failed(applyPartialConversion(module, target, std::move(patterns))))
            signalPassFailure();
    }
};

} // namespace

std::unique_ptr<Pass> createTlaLowerMutexToStdPass()
{
    return std::make_unique<TlaLowerMutexToStdPass>();
}

void registerTlaLowerMutexToStdPass()
{
    PassRegistration<TlaLowerMutexToStdPass>();
}

} // namespace tla
