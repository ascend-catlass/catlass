#include "PassesCommon.h"
#include "PassesInternal.h"
#include "Passes/TlaTensorToMemref.h"

// tla-finalize-memref: assert ABI/tensor lowering invariants and DCE dead
// scaffolding left by the region passes.

namespace tla {
namespace {

static LogicalResult validateNoTensorFunctionAbi(ModuleOp module)
{
    for (func::FuncOp funcOp : module.getOps<func::FuncOp>()) {
        FunctionType type = funcOp.getFunctionType();
        if (llvm::any_of(type.getInputs(), [](Type t) { return isa<::tla::TlaTensorType>(t); }) ||
            llvm::any_of(type.getResults(), [](Type t) { return isa<::tla::TlaTensorType>(t); }))
            return funcOp.emitError("tla.tensor remained on func.func ABI after lower-func");
    }
    LogicalResult result = success();
    module.walk([&](func::CallOp callOp) {
        if (llvm::any_of(callOp.getOperandTypes(), [](Type t) { return isa<::tla::TlaTensorType>(t); }) ||
            llvm::any_of(callOp.getResultTypes(), [](Type t) { return isa<::tla::TlaTensorType>(t); })) {
            callOp.emitError("tensor-typed func.call remained after lowering");
            result = failure();
        }
    });
    return result;
}

static bool hasNoResultUses(Operation* op)
{
    return llvm::all_of(op->getResults(), [](Value result) { return result.use_empty(); });
}

static bool isTlaTensorBridgeCast(UnrealizedConversionCastOp op)
{
    return llvm::any_of(op->getOperandTypes(), [](Type type) { return succeeded(decodeTensorTypeInfo(type)); }) ||
           llvm::any_of(op->getResultTypes(), [](Type type) { return succeeded(decodeTensorTypeInfo(type)); });
}

static bool isDeadTensorBridgeCast(UnrealizedConversionCastOp op)
{
    return hasNoResultUses(op.getOperation()) && isTlaTensorBridgeCast(op);
}

static void eraseDeadMaterializations(ModuleOp module)
{
    bool progress = true;
    while (progress) {
        progress = false;
        SmallVector<Operation*, 8> toErase;
        module.walk([&](Operation* op) {
            if (!hasNoResultUses(op))
                return;
            if (llvm::isa<mlir::memref::SubViewOp>(op)) {
                toErase.push_back(op);
                return;
            }
            if (llvm::isa<hivm::PointerCastOp>(op))
                toErase.push_back(op);
        });
        for (Operation* op : toErase) {
            if (!op->getBlock())
                continue;
            op->erase();
            progress = true;
        }
    }
}

struct EraseDeadTensorBridgeCastPattern : public OpRewritePattern<UnrealizedConversionCastOp> {
    using OpRewritePattern<UnrealizedConversionCastOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(UnrealizedConversionCastOp op, PatternRewriter& rewriter) const override
    {
        if (!isDeadTensorBridgeCast(op))
            return failure();
        rewriter.eraseOp(op);
        return success();
    }
};

template <typename OpT>
struct EraseDeadOpPattern : public OpRewritePattern<OpT> {
    using OpRewritePattern<OpT>::OpRewritePattern;

    LogicalResult matchAndRewrite(OpT op, PatternRewriter& rewriter) const override
    {
        if (!hasNoResultUses(op.getOperation()))
            return failure();
        rewriter.eraseOp(op);
        return success();
    }
};

class TlaFinalizeMemrefPass : public PassWrapper<TlaFinalizeMemrefPass, OperationPass<ModuleOp>> {
public:
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TlaFinalizeMemrefPass)

    StringRef getArgument() const override
    {
        return "tla-finalize-memref";
    }
    StringRef getName() const override
    {
        return "TlaFinalizeMemrefPass";
    }
    StringRef getDescription() const override
    {
        return "Finalize the tla.tensor -> memref lowering: DCE dead scaffolding and bridge casts.";
    }
    void getDependentDialects(DialectRegistry& registry) const override
    {
        registry.insert<
            arith::ArithDialect, func::FuncDialect, mlir::memref::MemRefDialect, scf::SCFDialect, hivm::HIVMDialect>();
    }

    void runOnOperation() override
    {
        ModuleOp module = getOperation();

        if (failed(validateNoTensorFunctionAbi(module))) {
            signalPassFailure();
            return;
        }

        ConversionTarget target(getContext());
        target.addLegalDialect<
            ::tla::TlaDialect, arith::ArithDialect, func::FuncDialect, mlir::memref::MemRefDialect, scf::SCFDialect,
            hivm::HIVMDialect>();
        target.markUnknownOpDynamicallyLegal([](Operation*) { return true; });
        // Assert the tensor / tile / region / compute ops this pass depends on being
        // gone are indeed gone. Pointer producers and transforms must already have
        // been eliminated by tla-lower-ptr; do not DCE them here, because that would
        // hide an upstream lowering failure. tla.tensor_desc is the exception: the
        // region passes materialize tile memrefs from it but may leave the op itself
        // dead, so it is DCE'd (not asserted). tla.mutex / tla.cross_* are
        // intentionally NOT asserted: they are lowered by
        // tla-lower-mutex-to-std / tla-lower-flag-barrier-to-hivm, which run AFTER
        // this pass, so they are still present when finalize runs.
        target.addIllegalOp<
            ::tla::TileViewOp, ::tla::ScalarLoadOp, ::tla::ScalarStoreOp, ::tla::CopyOp, ::tla::MakeTensorLikeOp,
            ::tla::MakeTensorOp, ::tla::LoadOp, ::tla::StoreOp, ::tla::LoadUpdateOp, ::tla::StoreUpdateOp,
            ::tla::LoadUnalignUpdateOp, ::tla::StoreUnalignUpdateOp, ::tla::LoadUnalignPreOp, ::tla::CreateUregOp,
            ::tla::StoreUnalignPostOp, ::tla::FuncOp, ::tla::ReturnOp, ::tla::CubeOp, ::tla::VectorOp, ::tla::MmadOp,
            ::tla::MmadMxOp, ::tla::FillOp, ::tla::AllocPtrOp, ::tla::RecastPtrOp, ::tla::TensorPtrOp,
            ::tla::PtrAddOp>();
        target.addDynamicallyLegalOp<
            ::tla::MakeShapeOp, ::tla::MakeCoordOp, ::tla::MakeStrideOp, ::tla::MakeLayoutOp, ::tla::IntToPtrOp,
            ::tla::PtrToIntOp, ::tla::TensorDescOp>([](Operation* op) { return !hasNoResultUses(op); });
        target.addDynamicallyLegalOp<UnrealizedConversionCastOp>(
            [](UnrealizedConversionCastOp op) { return !isDeadTensorBridgeCast(op); });
        target.addDynamicallyLegalOp<mlir::memref::SubViewOp>(
            [](mlir::memref::SubViewOp op) { return !hasNoResultUses(op.getOperation()); });
        target.addDynamicallyLegalOp<hivm::PointerCastOp>(
            [](hivm::PointerCastOp op) { return !hasNoResultUses(op.getOperation()); });

        for (int cleanupPass = 0; cleanupPass < 2; ++cleanupPass) {
            RewritePatternSet patterns(&getContext());
            patterns.add<
                EraseDeadTensorBridgeCastPattern, EraseDeadOpPattern<::tla::MakeShapeOp>,
                EraseDeadOpPattern<::tla::MakeCoordOp>, EraseDeadOpPattern<::tla::MakeStrideOp>,
                EraseDeadOpPattern<::tla::MakeLayoutOp>, EraseDeadOpPattern<::tla::IntToPtrOp>,
                EraseDeadOpPattern<::tla::PtrToIntOp>, EraseDeadOpPattern<mlir::memref::SubViewOp>,
                EraseDeadOpPattern<::tla::TensorDescOp>, EraseDeadOpPattern<hivm::PointerCastOp>>(&getContext());
            if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
                signalPassFailure();
                return;
            }
        }
        eraseDeadMaterializations(module);
        module.walk([](Operation* op) { op->removeAttr(kAllocSizeBytesMetadataAttrName); });
    }
};

} // namespace

std::unique_ptr<Pass> createTlaFinalizeMemrefPass()
{
    return std::make_unique<TlaFinalizeMemrefPass>();
}

void registerTlaFinalizeMemrefPass()
{
    PassRegistration<TlaFinalizeMemrefPass>();
}

} // namespace tla
