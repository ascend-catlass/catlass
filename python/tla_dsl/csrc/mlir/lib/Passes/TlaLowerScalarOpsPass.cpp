// Lower the TLA scalar ops that have no MLIR op of their own: the scalar unit's
// conversion instructions, data synchronization barriers, cache maintenance, CTRL
// and AR access.
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
// `replaceWithHelperCall` helper below are the extension point, so a new
// scalar op that maps to a helper is one entry plus the helper itself. The
// exception is tla.arch.dcci_cachelines, which becomes hivm.dcci directly: it
// carries no address, so there is nothing that op's optional memref operand
// would have to describe, and HIVM materializes the address-0 placeholder that
// the native entire-domain API passes.
// The AR special-register ops (tla.arch.get_ar and the scalar placement of
// tla.arch.clear_ar) arrive here for the same reason: ccec compiles them to
// llvm.hivm.GET.AR and llvm.hivm.sprclr, and neither intrinsic is registered in
// hivmc's LLVM. They differ from the conversions in where the helper lives --
// AR is a vector-unit register, so bc/Vector/ar_spr.cpp, declared AIV rather
// than AIC_OR_AIV.
//
// Addressed DCCI takes an i64 byte address from the existing pointer lowering.
// DCI uses an inline bitcode helper.

#include "PassesCommon.h"
#include "PassesInternal.h"

namespace tla {

namespace {

static void lowerCtrlAccess(ModuleOp module)
{
    module.walk([](ArchGetCtrlOp op) {
        OpBuilder builder(op);
        Location loc = op.getLoc();
        Value value = builder.create<hivm::GetCtrlInstrOp>(loc, builder.getI64Type());
        unsigned width = op.getHi() - op.getLo() + 1;
        if (width < 64) {
            Value shift = builder.create<arith::ConstantIntOp>(loc, op.getLo(), 64);
            Value mask = builder.create<arith::ConstantIntOp>(loc, (UINT64_C(1) << width) - 1, 64);
            value = builder.createOrFold<arith::ShRUIOp>(loc, value, shift);
            value = builder.createOrFold<arith::AndIOp>(loc, value, mask);
        }
        op.getValue().replaceAllUsesWith(value);
        op.erase();
    });
    module.walk([](ArchSetCtrlOp op) {
        OpBuilder builder(op);
        Location loc = op.getLoc();
        uint64_t mask = op.getMask();
        Value old = builder.create<hivm::GetCtrlInstrOp>(loc, builder.getI64Type());
        Value keep = builder.create<arith::ConstantIntOp>(loc, ~mask, 64);
        Value select = builder.create<arith::ConstantIntOp>(loc, mask, 64);
        Value preserved = builder.createOrFold<arith::AndIOp>(loc, old, keep);
        Value update = builder.createOrFold<arith::AndIOp>(loc, op.getBits(), select);
        Value merged = builder.createOrFold<arith::OrIOp>(loc, preserved, update);
        // Changing floating saturation requires flushing the atomic cache first.
        if (mask & (UINT64_C(1) << 48)) {
            builder.create<hivm::DCCIOp>(
                loc, hivm::DCCIMode::ALL_CACHE_LINES, hivm::DataCacheKind::ATOMIC, /*ptr=*/Value());
            builder.create<hivm::PipeBarrierOp>(loc, hivm::PipeAttr::get(builder.getContext(), hivm::PIPE::PIPE_ALL));
        }
        builder.create<hivm::SetCtrlInstrOp>(loc, merged);
        op.erase();
    });
}

// ALWAYS_INLINE keeps the linked helper from staying a real call. The core type
// is the helper's own: a scalar-unit instruction exists on both cores and is
// declared AIC_OR_AIV, while a helper that reaches vector-unit state lives in
// the vector bitcode only and must say AIV, so an AIC placement is a link error
// rather than a silently wrong result.
static func::FuncOp getOrCreateHelper(
    ModuleOp module, StringRef name, TypeRange operands, TypeRange results,
    hivm::TFuncCoreType coreType = hivm::TFuncCoreType::AIC_OR_AIV)
{
    if (auto existing = module.lookupSymbol<func::FuncOp>(name))
        return existing;
    OpBuilder builder(module.getBodyRegion());
    builder.setInsertionPointToStart(module.getBody());
    auto func = builder.create<func::FuncOp>(module.getLoc(), name, builder.getFunctionType(operands, results));
    func.setPrivate();
    MLIRContext* ctx = module.getContext();
    func->setAttr(hacc::stringifyEnum(hacc::HACCToLLVMIRTranslateAttr::ALWAYS_INLINE), UnitAttr::get(ctx));
    func->setAttr(hivm::TFuncCoreTypeAttr::name, hivm::TFuncCoreTypeAttr::get(ctx, coreType));
    return func;
}

// Replace one op with a call to `callee`, taking the op's operands and results.
template <typename OpTy>
static LogicalResult replaceWithHelperCall(
    OpTy op, StringRef callee, hivm::TFuncCoreType coreType = hivm::TFuncCoreType::AIC_OR_AIV)
{
    ModuleOp module = op->template getParentOfType<ModuleOp>();
    func::FuncOp declaration = getOrCreateHelper(module, callee, op->getOperandTypes(), op->getResultTypes(), coreType);
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

// tla.scalar_bit_op mode -> bitcode helper. All five queries take one uint64
// and return int64, so only the mode has to be mapped.
static StringRef scalarBitOpCallee(StringRef mode)
{
    return llvm::StringSwitch<StringRef>(mode)
        .Case("bcnt0", "_mlir_ciface_bcnt0")
        .Case("bcnt1", "_mlir_ciface_bcnt1")
        .Case("clz", "_mlir_ciface_clz")
        .Case("sff0", "_mlir_ciface_sff0")
        .Case("sff1", "_mlir_ciface_sff1")
        .Default(StringRef());
}

// tla.scalar_fast_div_divisor width -> bitcode helper pair. The two results
// are produced by two scalar helpers rather than one packed struct: a
// multi-result func.call lowers to a struct return, which does not survive the
// scalar-unit device ABI. The width selects both the algorithm (closed form vs
// software long division) and the helpers, so it cannot be collapsed.
static StringRef scalarFastDivShiftCallee(mlir::Type divisorType)
{
    if (divisorType.isInteger(32))
        return "_mlir_ciface_tla_fast_div_shift_32";
    if (divisorType.isInteger(64))
        return "_mlir_ciface_tla_fast_div_shift_64";
    return StringRef();
}

static StringRef scalarFastDivMagicCallee(mlir::Type divisorType)
{
    if (divisorType.isInteger(32))
        return "_mlir_ciface_tla_fast_div_magic_32";
    if (divisorType.isInteger(64))
        return "_mlir_ciface_tla_fast_div_magic_64";
    return StringRef();
}

// tla.scalar_simd_fast_{div,mod} width -> bitcode helper. One scalar return
// each, so the ordinary replaceWithHelperCall path applies; the SIMD side does
// not use a hardware mulhi, the helper computes it in software.
static StringRef scalarSimdFastDivCallee(mlir::Type dividendType)
{
    if (dividendType.isInteger(32))
        return "_mlir_ciface_tla_simd_fast_div_32";
    if (dividendType.isInteger(64))
        return "_mlir_ciface_tla_simd_fast_div_64";
    return StringRef();
}

static StringRef scalarSimdFastModCallee(mlir::Type dividendType)
{
    if (dividendType.isInteger(32))
        return "_mlir_ciface_tla_simd_fast_mod_32";
    if (dividendType.isInteger(64))
        return "_mlir_ciface_tla_simd_fast_mod_64";
    return StringRef();
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

// AR is a vector-unit register, and its helpers are in the vector meta-op
// bitcode only, so an occurrence on the cube core has nothing to resolve
// against. The frontend rejects a tla.cube() placement, and this covers
// hand-written IR: without it the failure surfaces as a link or "Unknown core
// type" error naming a helper the source never mentioned. Core types are
// settled by now -- tla-lower-func classifies each device function and
// tla-split-mixed-func has already split a mixed kernel in two.
static LogicalResult requireAivPlacement(Operation* op)
{
    auto func = op->getParentOfType<func::FuncOp>();
    if (!func)
        return success();
    auto coreType = func->getAttrOfType<hivm::TFuncCoreTypeAttr>(hivm::TFuncCoreTypeAttr::name);
    if (!coreType || coreType.getFuncCoreType() != hivm::TFuncCoreType::AIC)
        return success();
    return op->emitOpError("uses the vector unit's AR register and is unavailable on the AIC");
}

static LogicalResult lowerScalarICacheOps(ModuleOp module)
{
    SmallVector<Operation*> icacheOps;
    module.walk([&](Operation* op) {
        if (isa<ArchICachePrefetchOp, ArchGetICachePrefetchStatusOp>(op))
            icacheOps.push_back(op);
    });
    for (Operation* op : icacheOps) {
        if (!hasC310TargetAttrs(module)) {
            op->emitOpError("requires an Ascend 950 target");
            return failure();
        }
        auto func = op->getParentOfType<func::FuncOp>();
        auto issuer = func ? func->getAttrOfType<hivm::TFuncCoreTypeAttr>(hivm::TFuncCoreTypeAttr::name) : nullptr;
        if (!issuer || (issuer.getFuncCoreType() != hivm::TFuncCoreType::AIV &&
                        issuer.getFuncCoreType() != hivm::TFuncCoreType::AIC)) {
            op->emitOpError("requires a lowered scalar AIC or AIV function");
            return failure();
        }
        auto core = issuer.getFuncCoreType();
        bool isAiv = core == hivm::TFuncCoreType::AIV;
        if (auto prefetch = dyn_cast<ArchICachePrefetchOp>(op)) {
            OpBuilder builder(op);
            auto units =
                builder.create<arith::ConstantIntOp>(op->getLoc(), prefetch.getPrefetchUnitsAttr().getInt(), 64);
            StringRef callee = isAiv ? "_mlir_ciface_tla_icache_prefetch_aiv" : "_mlir_ciface_tla_icache_prefetch_aic";
            auto helper = getOrCreateHelper(module, callee, TypeRange{units.getType()}, {}, core);
            builder.create<func::CallOp>(op->getLoc(), helper, ValueRange{units.getResult()});
            op->erase();
        } else {
            StringRef callee = isAiv ? "_mlir_ciface_tla_get_icache_prefetch_status_aiv" :
                                       "_mlir_ciface_tla_get_icache_prefetch_status_aic";
            auto helper = getOrCreateHelper(module, callee, {}, op->getResultTypes(), core);
            OpBuilder builder(op);
            auto call = builder.create<func::CallOp>(op->getLoc(), helper, ValueRange{});
            op->replaceAllUsesWith(call.getResults());
            op->erase();
        }
    }
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
        return "Lower TLA scalar conversion, cache maintenance, CTRL, AR and trap operations";
    }

    void getDependentDialects(DialectRegistry& registry) const override
    {
        registry.insert<arith::ArithDialect, func::FuncDialect, hivm::HIVMDialect, LLVM::LLVMDialect>();
    }

    void runOnOperation() override
    {
        ModuleOp module = getOperation();
        if (failed(lowerScalarICacheOps(module)))
            return signalPassFailure();
        lowerCtrlAccess(module);

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
                if (failed(replaceWithHelperCall(op, dataSyncBarrierUbcallee())))
                    return signalPassFailure();
                continue;
            }
            StringRef callee = dataSyncBarrierCallee(domain);
            if (callee.empty()) {
                // The verifier rejects any other domain; this guards hand-written IR.
                op.emitOpError() << "unsupported data-sync-barrier domain \"" << domain << "\"";
                return signalPassFailure();
            }
            if (failed(replaceWithHelperCall(op, callee)))
                return signalPassFailure();
        }

        SmallVector<::tla::ArchLoadBypassDcacheOp, 8> bypassLoads;
        module.walk([&](::tla::ArchLoadBypassDcacheOp op) { bypassLoads.push_back(op); });
        for (::tla::ArchLoadBypassDcacheOp op : bypassLoads) {
            OpBuilder builder(op);
            Location loc = op.getLoc();
            auto pointerType =
                LLVM::LLVMPointerType::get(builder.getContext(), static_cast<unsigned>(hivm::AddressSpace::GM));
            Value pointer = builder.create<LLVM::IntToPtrOp>(loc, pointerType, op.getAddress());
            auto resultType = cast<IntegerType>(op.getResult().getType());
            auto signlessType = builder.getIntegerType(resultType.getWidth());
            Value result = builder.create<hivm::LoadScalarOp>(loc, signlessType, pointer);
            if (resultType != signlessType)
                result = builder.create<UnrealizedConversionCastOp>(loc, resultType, result).getResult(0);
            op.getResult().replaceAllUsesWith(result);
            op.erase();
        }

        SmallVector<::tla::ArchStoreBypassDcacheOp, 8> bypassStores;
        module.walk([&](::tla::ArchStoreBypassDcacheOp op) { bypassStores.push_back(op); });
        for (::tla::ArchStoreBypassDcacheOp op : bypassStores)
            if (failed(replaceWithHelperCall(op, "_mlir_ciface_tla_store_bypass_dcache_i32")))
                return signalPassFailure();

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
            if (failed(replaceWithHelperCall(op, "_mlir_ciface_tla_dci")))
                return signalPassFailure();
        }

        // Trap on the scalar unit. TrapOp::verify rejects the op inside
        // tla.vec.func, so every trap here is scalar code of a plain AIC/AIV
        // function. The two placement guards below stay as a backstop: this
        // `[aicore]` helper must never be emitted into an outlined vector
        // helper or a SIMT per-thread function, where the scalar trap would run
        // as vector code.
        SmallVector<::tla::TrapOp, 4> traps;
        module.walk([&](::tla::TrapOp op) { traps.push_back(op); });
        for (::tla::TrapOp op : traps) {
            func::FuncOp parentFunc = op->getParentOfType<func::FuncOp>();
            if (!parentFunc)
                return op.emitOpError("must be nested inside a kernel function"), signalPassFailure();
            if (parentFunc->hasAttr(kHivmVectorFunctionAttrName) ||
                parentFunc->hasAttr(hivm_regbaseintrins::kDavinciCallingConvAttrName))
                return op.emitOpError("was not lowered by its vector region pass"), signalPassFailure();
            std::optional<hivm::TFuncCoreType> coreType = getFunctionCoreType(parentFunc);
            if (coreType != hivm::TFuncCoreType::AIC && coreType != hivm::TFuncCoreType::AIV)
                return op.emitOpError("requires a split AIC or AIV function"), signalPassFailure();
            if (failed(replaceWithHelperCall(op, "_mlir_ciface_tla_trap")))
                return signalPassFailure();
        }

        // The AR read. AIV-only, so the helper is declared AIV and resolves out
        // of the vector meta-op bitcode alone.
        SmallVector<::tla::ArchGetArOp, 4> getArs;
        module.walk([&](::tla::ArchGetArOp op) { getArs.push_back(op); });
        for (::tla::ArchGetArOp op : getArs) {
            if (failed(requireAivPlacement(op)))
                return signalPassFailure();
            if (failed(replaceWithHelperCall(op, "_mlir_ciface_tla_get_ar", hivm::TFuncCoreType::AIV)))
                return signalPassFailure();
        }

        // The AR clear, in its scalar placement. An occurrence inside a SIMD
        // tla.vec.func is lowered by tla-vector-region-simd into the vector
        // helper it outlines and never reaches this pass; one that stays behind
        // -- written directly in tla.vector(), or flattened out of a vec.func
        // that had no tile work to outline -- needs a vector function of its
        // own, which is what this helper carries.
        SmallVector<::tla::ArchClearArOp, 4> clearArs;
        module.walk([&](::tla::ArchClearArOp op) { clearArs.push_back(op); });
        for (::tla::ArchClearArOp op : clearArs) {
            if (failed(requireAivPlacement(op)))
                return signalPassFailure();
            if (failed(replaceWithHelperCall(op, "_mlir_ciface_tla_clear_ar", hivm::TFuncCoreType::AIV)))
                return signalPassFailure();
        }

        SmallVector<::tla::ScalarBitOp, 8> bitOps;
        module.walk([&](::tla::ScalarBitOp op) { bitOps.push_back(op); });
        for (::tla::ScalarBitOp op : bitOps) {
            StringRef callee = scalarBitOpCallee(op.getMode());
            if (callee.empty()) {
                // The verifier rejects any other mode; this guards hand-written IR.
                op.emitOpError() << "unsupported bit query \"" << op.getMode() << "\"";
                return signalPassFailure();
            }
            if (failed(replaceWithHelperCall(op, callee)))
                return signalPassFailure();
        }

        SmallVector<::tla::ScalarFastDivDivisor, 8> fastDivDivisors;
        module.walk([&](::tla::ScalarFastDivDivisor op) { fastDivDivisors.push_back(op); });
        for (::tla::ScalarFastDivDivisor op : fastDivDivisors) {
            mlir::Type divisorType = op.getDivisor().getType();
            StringRef shiftCallee = scalarFastDivShiftCallee(divisorType);
            StringRef magicCallee = scalarFastDivMagicCallee(divisorType);
            if (shiftCallee.empty() || magicCallee.empty()) {
                // The verifier rejects a non-i32/i64 width; this guards hand-written IR.
                op.emitOpError() << "expects an i32 or i64 divisor, got " << divisorType;
                return signalPassFailure();
            }
            // Two scalar returns: shift first, then magic from (divisor, shift).
            func::FuncOp shiftDecl =
                getOrCreateHelper(module, shiftCallee, TypeRange{divisorType}, TypeRange{divisorType});
            func::FuncOp magicDecl =
                getOrCreateHelper(module, magicCallee, TypeRange{divisorType, divisorType}, TypeRange{divisorType});
            OpBuilder builder(op);
            auto shiftCall = builder.create<func::CallOp>(op.getLoc(), shiftDecl, ValueRange{op.getDivisor()});
            auto magicCall = builder.create<func::CallOp>(
                op.getLoc(), magicDecl, ValueRange{op.getDivisor(), shiftCall.getResult(0)});
            op.getMagic().replaceAllUsesWith(magicCall.getResult(0));
            op.getShift().replaceAllUsesWith(shiftCall.getResult(0));
            op.erase();
        }

        SmallVector<::tla::ScalarSimdFastDiv, 8> simdFastDivs;
        module.walk([&](::tla::ScalarSimdFastDiv op) { simdFastDivs.push_back(op); });
        for (::tla::ScalarSimdFastDiv op : simdFastDivs) {
            StringRef callee = scalarSimdFastDivCallee(op.getDividend().getType());
            if (callee.empty()) {
                // The verifier rejects a non-i32/i64 width; this guards hand-written IR.
                op.emitOpError() << "expects an i32 or i64 dividend, got " << op.getDividend().getType();
                return signalPassFailure();
            }
            if (failed(replaceWithHelperCall(op, callee)))
                return signalPassFailure();
        }

        SmallVector<::tla::ScalarSimdFastMod, 8> simdFastMods;
        module.walk([&](::tla::ScalarSimdFastMod op) { simdFastMods.push_back(op); });
        for (::tla::ScalarSimdFastMod op : simdFastMods) {
            StringRef callee = scalarSimdFastModCallee(op.getDividend().getType());
            if (callee.empty()) {
                op.emitOpError() << "expects an i32 or i64 dividend, got " << op.getDividend().getType();
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
