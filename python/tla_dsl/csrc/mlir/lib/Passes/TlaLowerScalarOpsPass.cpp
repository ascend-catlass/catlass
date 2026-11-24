// Lower the TLA scalar ops that have no MLIR op of their own: the scalar unit's
// conversion instructions, the data synchronization barrier (one dsb per
// storage domain; the ub-domain helper is AIV-only, see below), and the scalar
// architectural state ops -- the three data cache maintenance operations behind
// tla.arch.
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
// Two routes go down from here, because HIVM covers only part of what these ops
// need. Most become a call to a meta-op bitcode helper; the table and the
// `replaceWithScalarHelperCall` helper below are the extension point, so a new
// scalar op that maps to a helper is one entry plus the helper itself. The
// exception is tla.arch.dcci_cachelines, which becomes hivm.dcci directly: it
// carries no address, so there is nothing that op's optional memref operand
// would have to describe, and HIVM materializes the address-0 placeholder that
// the native entire-domain API passes.
//
// Addressed DCCI takes an i64 byte address from the existing pointer lowering.
// DCI uses an inline bitcode helper.

#include "PassesCommon.h"
#include "PassesInternal.h"

namespace tla {

namespace {

// A bitcode helper is reached through a private func.func declaration which the
// linker resolves against the per-core bitcode module. The declaration carries
// two attributes, and both matter: ALWAYS_INLINE keeps the linked helper from
// staying a real call, and AIC_OR_AIV says the scalar unit it targets exists on
// both cores, so one declaration serves an AIC and an AIV function alike.
//
// Declare `name` in `module`, or return the declaration already there.
func::FuncOp getOrCreateScalarHelper(ModuleOp module, StringRef name, TypeRange operands, TypeRange results)
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
LogicalResult replaceWithScalarHelperCall(OpTy op, StringRef callee)
{
    ModuleOp module = op->template getParentOfType<ModuleOp>();
    func::FuncOp declaration = getOrCreateScalarHelper(module, callee, op->getOperandTypes(), op->getResultTypes());
    OpBuilder builder(op);
    auto call = builder.create<func::CallOp>(op.getLoc(), declaration, op->getOperands());
    op->replaceAllUsesWith(call.getResults());
    op->erase();
    return success();
}

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

// tla.data_sync_barrier storage domain -> bitcode helper. One no-arg helper
// per domain rather than one helper plus a runtime switch: ALWAYS_INLINE makes
// every call site exactly one dsb instruction, with no branch and no parameter
// passing. The helpers must not be marked pure, or CSE/DCE could merge or drop
// repeated barriers. The helper symbols mirror the DSL domain names.
static StringRef dataSyncBarrierCallee(StringRef domain)
{
    return llvm::StringSwitch<StringRef>(domain)
        .Case("gm_ub", "_mlir_ciface_tla_data_sync_barrier_gm_ub")
        .Case("gm", "_mlir_ciface_tla_data_sync_barrier_gm")
        .Case("seq", "_mlir_ciface_tla_data_sync_barrier_seq")
        .Default(StringRef());
}

// The ub domain needs a core-specific symbol: hivmc links BOTH meta_op.aic and
// meta_op.aiv into each subcore compile, and the AIC-compiled ub helper has an
// empty body (DataSyncBarrierImpl returns early for MemDsbT::UB on AIC) while
// the AIV-compiled one issues the dsb. A shared name would let the AIC body win
// on AIV and silently drop the barrier -- the same hazard that split
// _mlir_ciface_tla_init_soc_state into _aic/_aiv (TlaPrologueEpiloguePass).
// The AIV-side helper therefore lives in bc/Vector/ (compiled only into
// meta_op.aiv.bc) as _mlir_ciface_tla_data_sync_barrier_ub_aiv, and AIC
// functions drop the op entirely: dsb<ub> is a hardware no-op there, which is
// exactly what the frontend documents for mixed-kernel scalar bodies.
static StringRef dataSyncBarrierUbcallee()
{
    return "_mlir_ciface_tla_data_sync_barrier_ub_aiv";
}

// tla.arch.dcci_cachelines target -> HIVM data cache kind. The op's verifier
// rejects `ub`, so only the three domains the DSL exposes can reach this; HIVM
// itself has the kind either way, which is why the guard below is a diagnostic
// and not an unrepresentable case.
static std::optional<hivm::DataCacheKind> dcciDataCacheKind(::DcciTarget target)
{
    switch (target) {
        case ::DcciTarget::all:
            return hivm::DataCacheKind::ALL;
        case ::DcciTarget::gm_l2:
            return hivm::DataCacheKind::OUT;
        case ::DcciTarget::st_atomic:
            return hivm::DataCacheKind::ATOMIC;
        case ::DcciTarget::ub:
            break;
    }
    return std::nullopt;
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
        return "Lower TLA scalar conversion and cache control operations";
    }

    void getDependentDialects(DialectRegistry& registry) const override
    {
        registry.insert<arith::ArithDialect, func::FuncDialect, hivm::HIVMDialect, LLVM::LLVMDialect>();
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
            if (failed(replaceWithScalarHelperCall(op, callee)))
                return signalPassFailure();
        }

        SmallVector<::tla::DataSyncBarrierOp, 8> dataSyncBarriers;
        module.walk([&](::tla::DataSyncBarrierOp op) { dataSyncBarriers.push_back(op); });
        for (::tla::DataSyncBarrierOp op : dataSyncBarriers) {
            StringRef domain = stringifyDsbTarget(op.getDomain());
            if (domain == "ub") {
                // dsb<ub> is a hardware no-op on AIC (no UB there); drop it
                // instead of emitting an empty call. Everything else -- AIV,
                // and defensively MIX/unknown types, where the barrier is the
                // meaningful form -- calls the AIV-side helper. See the
                // dataSyncBarrierUbcallee comment for why the symbol is
                // core-specific.
                func::FuncOp parentFunc = op->getParentOfType<func::FuncOp>();
                std::optional<hivm::TFuncCoreType> coreType = getFunctionCoreType(parentFunc);
                if (coreType == hivm::TFuncCoreType::AIC) {
                    op->erase();
                    continue;
                }
                if (failed(replaceWithScalarHelperCall(op, dataSyncBarrierUbcallee())))
                    return signalPassFailure();
                continue;
            }
            StringRef callee = dataSyncBarrierCallee(domain);
            if (callee.empty()) {
                // The verifier rejects any other domain; this guards hand-written IR.
                op.emitOpError() << "unsupported data-sync-barrier domain \"" << domain << "\"";
                return signalPassFailure();
            }
            if (failed(replaceWithScalarHelperCall(op, callee)))
                return signalPassFailure();
        }

        SmallVector<::tla::ArchDcciCachelineOp, 8> cachelines;
        module.walk([&](::tla::ArchDcciCachelineOp op) { cachelines.push_back(op); });
        for (::tla::ArchDcciCachelineOp op : cachelines) {
            OpBuilder builder(op);
            Location loc = op.getLoc();
            auto pointerType =
                LLVM::LLVMPointerType::get(builder.getContext(), static_cast<unsigned>(hivm::AddressSpace::GM));
            Value pointer = builder.create<LLVM::IntToPtrOp>(loc, pointerType, op.getAddress());
            Value mode =
                builder.create<arith::ConstantIntOp>(loc, static_cast<int64_t>(hivm::DCCIMode::SINGLE_CACHE_LINE), 64);
            Value kind = builder.create<arith::ConstantIntOp>(loc, static_cast<int64_t>(hivm::DataCacheKind::ALL), 64);
            builder.create<hivm::DCCIDstInstrOp>(loc, pointer, mode, kind);
            op.erase();
        }

        // One whole dcache domain. This one needs no bitcode: with no address to
        // pass there is nothing for hivm.dcci's memref operand to describe.
        SmallVector<::tla::ArchDcciCachelinesOp, 8> dcciDomains;
        module.walk([&](::tla::ArchDcciCachelinesOp op) { dcciDomains.push_back(op); });
        for (::tla::ArchDcciCachelinesOp op : dcciDomains) {
            std::optional<hivm::DataCacheKind> kind = dcciDataCacheKind(op.getTarget());
            if (!kind) {
                // The verifier rejects `ub`; this guards hand-written IR.
                op.emitOpError() << "unsupported target \"" << ::stringifyDcciTarget(op.getTarget()) << "\"";
                return signalPassFailure();
            }
            OpBuilder builder(op);
            builder.create<hivm::DCCIOp>(op.getLoc(), hivm::DCCIMode::ALL_CACHE_LINES, *kind, /*ptr=*/Value());
            op->erase();
        }

        // Whole-dcache invalidate without writeback.
        SmallVector<::tla::ArchDciOp, 8> dciOps;
        module.walk([&](::tla::ArchDciOp op) { dciOps.push_back(op); });
        for (::tla::ArchDciOp op : dciOps) {
            if (failed(replaceWithScalarHelperCall(op, "_mlir_ciface_tla_dci")))
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
