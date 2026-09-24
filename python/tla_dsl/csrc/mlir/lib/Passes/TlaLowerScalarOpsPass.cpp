// Lower the TLA scalar ops that have no MLIR op of their own into calls to the
// meta-op bitcode helpers that carry the matching scalar instruction.
//
// Why these need a lowering at all, when tla.sqrt does not: math.sqrt and
// math.absf are ordinary MLIR ops the backend selects, so the frontend emits
// them directly. The scalar unit's conversion instructions have no such op, and
// neither shorter route reaches them:
//
//   * math.floor / ceil / round / roundeven and llvm.intr.rint compile, but
//     leave an undefined floorf / ceilf / roundf / roundevenf / rintf symbol --
//     LLVM lowers them to a libm call rather than to the instruction. Note that
//     hivmc still PRODUCES an object file for these, so "it compiled" is not the
//     signal; the undefined symbol is.
//   * llvm.call_intrinsic "llvm.hivm.CONV.f322s32r" is rejected by hivmc with
//     "could not find LLVM intrinsic": the CONV intrinsics are registered in the
//     CCE clang, not in the LLVM hivmc links against.
//
// ccec does select them from bc/Scalar/scalar_round_cast.cpp, so a call to the
// always_inline helper there is the one route that works, and costs a single
// instruction once the bitcode is linked.
//
// The table below is the extension point: a new scalar op that maps to a bitcode
// helper is one entry plus the helper itself, not a new pass.
//
// Placement (see buildTlaPipeline): after tla-lower-ptr, so an op taking an
// address sees an i64 rather than a !tla.ptr -- the atomics and the D-cache
// bypass store will need that -- and downstream of the region passes, so that
// anything which might SYNTHESIZE a scalar op still has a lowering behind it.
// tla-vector-region already synthesizes instruction sequences of its own, so
// that is the constraint worth designing to; it is not a correctness
// requirement for the ops here today, since a func.call survives
// vector-function outlining, as tla-lower-extern-call demonstrates from much
// earlier in the pipeline.

#include "PassesCommon.h"
#include "PassesInternal.h"

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"

namespace tla {

namespace {

// tla.scalar_round_cast mode -> bitcode helper. The mode/result-type pairing is
// enforced by the op's verifier, so this only has to map a validated mode.
static StringRef scalarRoundCastCallee(StringRef mode)
{
    return llvm::StringSwitch<StringRef>(mode)
        .Case("rn", "_mlir_ciface_tla_cast_f32_to_i32_rn")
        .Case("ra", "_mlir_ciface_tla_cast_f32_to_i32_ra")
        .Case("rd", "_mlir_ciface_tla_cast_f32_to_i32_rd")
        .Case("ru", "_mlir_ciface_tla_cast_f32_to_i32_ru")
        .Case("o", "_mlir_ciface_tla_cast_f32_to_f16_o")
        .Default(StringRef());
}

// The scalar unit is on both cores, so the declaration is AIC_OR_AIV;
// ALWAYS_INLINE keeps the linked helper from staying a real call.
static func::FuncOp getOrCreateHelper(ModuleOp module, StringRef name, TypeRange operands, TypeRange results)
{
    if (auto existing = module.lookupSymbol<func::FuncOp>(name))
        return existing;
    OpBuilder builder(module.getBodyRegion());
    builder.setInsertionPointToStart(module.getBody());
    auto func = builder.create<func::FuncOp>(module.getLoc(), name, builder.getFunctionType(operands, results));
    func.setPrivate();
    MLIRContext* ctx = module.getContext();
    func->setAttr(hacc::stringifyEnum(hacc::HACCToLLVMIRTranslateAttr::ALWAYS_INLINE), UnitAttr::get(ctx));
    func->setAttr(hivm::TFuncCoreTypeAttr::name, hivm::TFuncCoreTypeAttr::get(ctx, hivm::TFuncCoreType::AIC_OR_AIV));
    return func;
}

// Replace one op with a call to `callee`, taking the op's operands and results.
template <typename OpTy>
static LogicalResult replaceWithHelperCall(OpTy op, StringRef callee)
{
    ModuleOp module = op->template getParentOfType<ModuleOp>();
    func::FuncOp declaration = getOrCreateHelper(module, callee, op->getOperandTypes(), op->getResultTypes());
    OpBuilder builder(op);
    auto call = builder.create<func::CallOp>(op.getLoc(), declaration, op->getOperands());
    op->replaceAllUsesWith(call.getResults());
    op->erase();
    return success();
}

class TlaLowerScalarOpsPass : public PassWrapper<TlaLowerScalarOpsPass, OperationPass<ModuleOp>> {
public:
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TlaLowerScalarOpsPass)

    StringRef getArgument() const override
    {
        return "tla-lower-scalar-ops";
    }
    StringRef getName() const override
    {
        return "TlaLowerScalarOpsPass";
    }
    StringRef getDescription() const override
    {
        return "Lower TLA scalar ops to calls into the meta-op bitcode helpers";
    }

    void getDependentDialects(DialectRegistry& registry) const override
    {
        registry.insert<func::FuncDialect, hivm::HIVMDialect, LLVM::LLVMDialect>();
    }

    void runOnOperation() override
    {
        ModuleOp module = getOperation();

        SmallVector<::tla::ScalarRoundCastOp, 8> roundCasts;
        module.walk([&](::tla::ScalarRoundCastOp op) { roundCasts.push_back(op); });
        for (::tla::ScalarRoundCastOp op : roundCasts) {
            StringRef callee = scalarRoundCastCallee(op.getMode());
            if (callee.empty()) {
                // The verifier rejects any other mode; this guards hand-written IR.
                op.emitOpError() << "unsupported rounding mode \"" << op.getMode() << "\"";
                return signalPassFailure();
            }
            if (failed(replaceWithHelperCall(op, callee)))
                return signalPassFailure();
        }
    }
};

} // namespace

std::unique_ptr<Pass> createTlaLowerScalarOpsPass()
{
    return std::make_unique<TlaLowerScalarOpsPass>();
}

void registerTlaLowerScalarOpsPass()
{
    PassRegistration<TlaLowerScalarOpsPass>();
}

} // namespace tla
