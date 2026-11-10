#include "PassesCommon.h"

namespace tla {
namespace {

// CTRL[51] is the mmad M/N compute-direction priority bit.
static constexpr unsigned int ComputeOrderBit = 51;

// CTRL[46] is the mmad HF32 rounding-mode enable bit.
static constexpr unsigned int HF32ModeBit = 46;

// CTRL[47] is the mmad HF32 rounding-mode select bit.
static constexpr unsigned int HF32TransModeBit = 47;

// Call BC helpers by ``_mlir_ciface_*`` directly (same pattern as
// ``tla-lower-scalar-ops``). AIC and AIV must use distinct symbols: hivmc links
// both meta_op.aic and meta_op.aiv into each subcore via
// ``--link-aicore-bitcode``, so a shared name would let the cube body
// (``set_padding``) win and run on AIV → scalar instruction abnormal.
static constexpr StringLiteral kInitSocStateCalleeAic = "_mlir_ciface_tla_init_soc_state_aic";
static constexpr StringLiteral kInitSocStateCalleeAiv = "_mlir_ciface_tla_init_soc_state_aiv";

static func::FuncOp getOrCreateInitSocStateDecl(ModuleOp module, StringLiteral callee, hivm::TFuncCoreType coreType)
{
    if (auto existing = module.lookupSymbol<func::FuncOp>(callee))
        return existing;

    OpBuilder builder(module.getBodyRegion());
    builder.setInsertionPointToStart(module.getBody());
    auto func =
        builder.create<func::FuncOp>(module.getLoc(), callee, builder.getFunctionType(/*inputs=*/{}, /*results=*/{}));
    func.setPrivate();
    MLIRContext* ctx = module.getContext();
    func->setAttr(hacc::stringifyEnum(hacc::HACCToLLVMIRTranslateAttr::ALWAYS_INLINE), UnitAttr::get(ctx));
    func->setAttr(hivm::TFuncCoreTypeAttr::name, hivm::TFuncCoreTypeAttr::get(ctx, coreType));
    return func;
}

class TlaPrologueEpiloguePass : public PassWrapper<TlaPrologueEpiloguePass, OperationPass<ModuleOp>> {
public:
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TlaPrologueEpiloguePass)

    StringRef getArgument() const override
    {
        return "tla-prologue-epilogue";
    }
    StringRef getName() const override
    {
        return "TlaPrologueEpiloguePass";
    }
    StringRef getDescription() const override
    {
        return "Add HIVM kernel prologue and epilogue operations.";
    }

    void runOnOperation() override
    {
        MLIRContext* ctx = &getContext();
        OpBuilder builder(ctx);
        auto pipeAll = hivm::PipeAttr::get(ctx, hivm::PIPE::PIPE_ALL);
        ModuleOp module = getOperation();

        for (func::FuncOp funcOp : module.getOps<func::FuncOp>()) {
            if (funcOp.isDeclaration())
                continue;
            if (!hasRequiredHaccEntryAttrs(funcOp))
                continue;

            Block& entry = funcOp.getBody().front();
            Location loc = funcOp.getLoc();
            std::optional<hivm::TFuncCoreType> coreType = getFunctionCoreType(funcOp);
            if (!coreType) {
                funcOp.emitError("HACC entry is missing hivm.func_core_type");
                return signalPassFailure();
            }
            // Declare only the callee this entry needs. Emitting both AIC and
            // AIV private decls into an AIV-only module makes
            // ``_resolve_compile_option_from_lowered_mlir`` see an AIC
            // ``func_core_type`` and wrongly select meta_op.aic bitcode.
            bool isAiv = *coreType == hivm::TFuncCoreType::AIV;
            StringLiteral callee = isAiv ? kInitSocStateCalleeAiv : kInitSocStateCalleeAic;
            func::FuncOp initSoc = getOrCreateInitSocStateDecl(module, callee, *coreType);

            // Kernel entry: InitSocState only (no extra CTRL[60]/[48]/[56] writes).
            builder.setInsertionPointToStart(&entry);
            builder.create<func::CallOp>(loc, initSoc, ValueRange{});

            Operation* terminator = entry.getTerminator();
            Operation* lastBodyOp = terminator ? terminator->getPrevNode() : nullptr;
            if (auto barrier = llvm::dyn_cast_or_null<hivm::PipeBarrierOp>(lastBodyOp)) {
                if (barrier.getPipe().getPipe() == hivm::PIPE::PIPE_ALL) {
                    // Already has a trailing PIPE_ALL barrier: restore CTRL[51] / CTRL[46] /
                    // CTRL[47] to the hardware defaults just before it.
                    builder.setInsertionPoint(barrier);
                    builder.create<hivm::SetCtrlOp>(loc, /*enable=*/false, ComputeOrderBit);
                    builder.create<hivm::SetCtrlOp>(loc, /*enable=*/false, HF32ModeBit);
                    builder.create<hivm::SetCtrlOp>(loc, /*enable=*/false, HF32TransModeBit);
                    continue;
                }
            }

            if (terminator)
                builder.setInsertionPoint(terminator);
            else
                builder.setInsertionPointToEnd(&entry);
            // Restore CTRL[51] / CTRL[46] / CTRL[47] to the hardware defaults, then add
            // the trailing barrier.
            builder.create<hivm::SetCtrlOp>(loc, /*enable=*/false, ComputeOrderBit);
            builder.create<hivm::SetCtrlOp>(loc, /*enable=*/false, HF32ModeBit);
            builder.create<hivm::SetCtrlOp>(loc, /*enable=*/false, HF32TransModeBit);
            builder.create<hivm::PipeBarrierOp>(loc, pipeAll);
        }
    }
};

} // namespace

std::unique_ptr<Pass> createTlaPrologueEpiloguePass()
{
    return std::make_unique<TlaPrologueEpiloguePass>();
}

void registerTlaPrologueEpiloguePass()
{
    PassRegistration<TlaPrologueEpiloguePass>();
}

} // namespace tla
