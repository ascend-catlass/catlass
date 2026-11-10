#include "Passes/TlaVectorRegionCommon.h"
#include "PassesCommon.h"
#include "PassesInternal.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/SymbolTable.h"

namespace tla {
namespace {

static bool isVectorCall(::tla::CallExternOp op)
{
    return op->getParentOfType<::tla::VectorOp>() ||
           getFunctionCoreType(op->getParentOfType<func::FuncOp>()) == hivm::TFuncCoreType::AIV;
}

// Validate that a SIMD VF with an extern call has either tensor memory access or extern pointer argument.
static LogicalResult verifyExternMemorySource(::tla::VecFuncOp vf)
{
    if (!vecregion::isSimdVecFunc(vf))
        return success();
    bool hasExtern = false;
    auto walkResult = vf.walk([&](Operation* nested) {
        if (auto call = dyn_cast<::tla::CallExternOp>(nested)) {
            hasExtern = true;
            if (llvm::any_of(call.getOperandTypes(), [](Type t) { return isa<::tla::PtrType>(t); }))
                return WalkResult::interrupt();
        }
        if (isa<::tla::LoadOp, ::tla::StoreOp, ::tla::LoadUpdateOp, ::tla::StoreUpdateOp, ::tla::LoadUnalignUpdateOp,
                ::tla::StoreUnalignUpdateOp, ::tla::ScalarLoadOp, ::tla::ScalarStoreOp, ::tla::CopyOp>(nested))
            return WalkResult::interrupt();
        if (auto gather = dyn_cast<::tla::GatherOp>(nested);
            gather && isa<::tla::TlaTensorType>(gather.getX().getType()))
            return WalkResult::interrupt();
        return WalkResult::advance();
    });
    const bool hasMemorySource = walkResult.wasInterrupted();
    if (hasExtern && !hasMemorySource)
        return vf.emitOpError(
            "SIMD VF with an extern call requires a tensor memory access or an extern pointer argument");
    return success();
}

static void updateCallCoreType(func::FuncOp callee, bool isVector)
{
    MLIRContext* ctx = callee.getContext();
    hivm::TFuncCoreType callerCoreType = isVector ? hivm::TFuncCoreType::AIV : hivm::TFuncCoreType::AIC;
    auto calleeCoreType = callee->getAttrOfType<hivm::TFuncCoreTypeAttr>(hivm::TFuncCoreTypeAttr::name);
    if (!calleeCoreType) {
        callee->setAttr(hivm::TFuncCoreTypeAttr::name, hivm::TFuncCoreTypeAttr::get(ctx, callerCoreType));
        return;
    }
    if (calleeCoreType.getFuncCoreType() != callerCoreType) {
        callee->setAttr(
            hivm::TFuncCoreTypeAttr::name, hivm::TFuncCoreTypeAttr::get(ctx, hivm::TFuncCoreType::AIC_OR_AIV));
    }
}

// Lower tla.call_extern outside vf to a private func.func declaration + func.call.
class TlaLowerExternCallPass : public PassWrapper<TlaLowerExternCallPass, OperationPass<ModuleOp>> {
public:
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TlaLowerExternCallPass)

    StringRef getArgument() const override
    {
        return "tla-lower-extern-call";
    }
    StringRef getName() const override
    {
        return "TlaLowerExternCallPass";
    }
    StringRef getDescription() const override
    {
        return "Declare extern functions and lower calls outside SIMD VFs";
    }
    void getDependentDialects(DialectRegistry& registry) const override
    {
        registry.insert<func::FuncDialect, ::tla::TlaDialect>();
    }

    void runOnOperation() override
    {
        ModuleOp module = getOperation();
        if (module
                .walk([&](::tla::VecFuncOp vf) {
                    return failed(verifyExternMemorySource(vf)) ? WalkResult::interrupt() : WalkResult::advance();
                })
                .wasInterrupted()) {
            signalPassFailure();
            return;
        }
        SmallVector<::tla::CallExternOp, 4> calls;
        module.walk([&](::tla::CallExternOp op) { calls.push_back(op); });

        for (::tla::CallExternOp op : calls) {
            StringRef symbol = op.getCallee();
            bool isVector = isVectorCall(op);
            SmallVector<Type, 4> operandTypes(op.getOperandTypes());
            auto functionType = FunctionType::get(module.getContext(), operandTypes, op.getResultTypes());
            func::FuncOp callee = module.lookupSymbol<func::FuncOp>(symbol);
            if (!callee) {
                OpBuilder moduleBuilder(module.getBodyRegion());
                moduleBuilder.setInsertionPointToStart(module.getBody());
                callee = moduleBuilder.create<func::FuncOp>(op.getLoc(), symbol, functionType);
                callee.setPrivate();
            } else if (!callee.isDeclaration()) {
                op.emitOpError() << "external symbol @" << symbol << " conflicts with a defined function";
                signalPassFailure();
                return;
            } else if (callee.getFunctionType() != functionType) {
                op.emitOpError() << "external symbol @" << symbol << " was called with incompatible function types";
                signalPassFailure();
                return;
            }
            updateCallCoreType(callee, isVector);

            // SIMD outlining emits the call inside the generated VF function.
            if (op->getParentOfType<::tla::VecFuncOp>())
                continue;
            OpBuilder builder(op);
            builder.create<func::CallOp>(op.getLoc(), callee, op.getOperands());
            op.erase();
        }
    }
};

} // namespace

std::unique_ptr<Pass> createTlaLowerExternCallPass()
{
    return std::make_unique<TlaLowerExternCallPass>();
}

void registerTlaLowerExternCallPass()
{
    PassRegistration<TlaLowerExternCallPass>();
}

} // namespace tla
