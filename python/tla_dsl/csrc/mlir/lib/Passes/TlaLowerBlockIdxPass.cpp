#include "Passes/TlaAtomicCommon.h"
#include "PassesCommon.h"

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"

namespace tla {
namespace {

struct LowerBlockIdxOp : public OpRewritePattern<::tla::BlockIdxOp> {
    using OpRewritePattern<::tla::BlockIdxOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(::tla::BlockIdxOp op, PatternRewriter& rewriter) const override
    {
        auto loc = op.getLoc();
        auto hivmOp = rewriter.create<hivm::GetBlockIdxOp>(loc, rewriter.getI64Type());
        // HIVM returns i64; dialect / user model use i32.
        auto i32Value = rewriter.create<arith::TruncIOp>(loc, rewriter.getI32Type(), hivmOp.getResult());
        rewriter.replaceOp(op, i32Value.getResult());
        return success();
    }
};

struct LowerSubBlockIdxOp : public OpRewritePattern<::tla::SubBlockIdxOp> {
    using OpRewritePattern<::tla::SubBlockIdxOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(::tla::SubBlockIdxOp op, PatternRewriter& rewriter) const override
    {
        auto loc = op.getLoc();
        auto hivmOp = rewriter.create<hivm::GetSubBlockIdxOp>(loc, rewriter.getI64Type());
        auto i32Value = rewriter.create<arith::TruncIOp>(loc, rewriter.getI32Type(), hivmOp.getResult());
        rewriter.replaceOp(op, i32Value.getResult());
        return success();
    }
};

struct LowerSubBlockNumOp : public OpRewritePattern<::tla::SubBlockNumOp> {
    using OpRewritePattern<::tla::SubBlockNumOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(::tla::SubBlockNumOp op, PatternRewriter& rewriter) const override
    {
        auto coreType = getFunctionCoreType(op->getParentOfType<func::FuncOp>());
        if (coreType == hivm::TFuncCoreType::AIC)
            rewriter.replaceOpWithNewOp<arith::ConstantIntOp>(op, 1, 64);
        else if (coreType == hivm::TFuncCoreType::AIV)
            rewriter.replaceOpWithNewOp<hivm::GetSubBlockNumOp>(op, rewriter.getI64Type());
        else
            return op.emitOpError("requires an enclosing device function with AIC or AIV core type");
        return success();
    }
};

struct LowerBlockNumOp : public OpRewritePattern<::tla::BlockNumOp> {
    using OpRewritePattern<::tla::BlockNumOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(::tla::BlockNumOp op, PatternRewriter& rewriter) const override
    {
        auto loc = op.getLoc();
        auto hivmOp = rewriter.create<hivm::GetBlockNumOp>(loc, rewriter.getI64Type());
        auto i32Value = rewriter.create<arith::TruncIOp>(loc, rewriter.getI32Type(), hivmOp.getResult());
        rewriter.replaceOp(op, i32Value.getResult());
        return success();
    }
};

struct LowerGetSystemCycleOp : public OpRewritePattern<::tla::GetSystemCycleOp> {
    using OpRewritePattern<::tla::GetSystemCycleOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(::tla::GetSystemCycleOp op, PatternRewriter& rewriter) const override
    {
        auto module = op->getParentOfType<ModuleOp>();
        auto caller = op->getParentOfType<func::FuncOp>();
        hivm::TFuncCoreTypeAttr core;
        if (caller)
            core = caller->getAttrOfType<hivm::TFuncCoreTypeAttr>(hivm::TFuncCoreTypeAttr::name);
        if (!core ||
            (core.getFuncCoreType() != hivm::TFuncCoreType::AIC && core.getFuncCoreType() != hivm::TFuncCoreType::AIV))
            return op.emitOpError("requires a split AIC or AIV function");
        StringRef name =
            core.getFuncCoreType() == hivm::TFuncCoreType::AIV ? "tla_get_system_cycle.vector" : "tla_get_system_cycle";
        Operation* symbol = SymbolTable::lookupSymbolIn(module, name);
        auto callee = dyn_cast_or_null<func::FuncOp>(symbol);
        if (symbol && !callee)
            return op.emitOpError("counter helper symbol is reserved for a function declaration");
        auto type = rewriter.getFunctionType({}, {rewriter.getI64Type()});
        if (callee && callee.getFunctionType() != type)
            return op.emitOpError("counter helper symbol has an incompatible type");
        if (callee && (!callee.isDeclaration() || !callee.isPrivate() ||
                       callee->getAttrOfType<hivm::TFuncCoreTypeAttr>(hivm::TFuncCoreTypeAttr::name) != core))
            return op.emitOpError("counter helper symbol is reserved for the issuing core's private declaration");
        if (!callee) {
            OpBuilder::InsertionGuard guard(rewriter);
            rewriter.setInsertionPointToStart(module.getBody());
            callee = rewriter.create<func::FuncOp>(op.getLoc(), name, type);
            callee.setPrivate();
            callee->setAttr(hivm::TFuncCoreTypeAttr::name, core);
        }
        auto call = rewriter.create<func::CallOp>(op.getLoc(), callee, ValueRange{});
        rewriter.replaceOp(op, call.getResult(0));
        return success();
    }
};

// tla-lower-ptr keeps a typed pointer alive at pointer consumers and rebuilds
// it as tla.inttoptr(i64). The atom is such a consumer, so peel that bridge back
// to the byte address the intrinsic takes.
static Value getAtomicAddress(Value ptr)
{
    if (auto intToPtr = ptr.getDefiningOp<::tla::IntToPtrOp>())
        return intToPtr.getAddr();
    return {};
}

// A same-width integer unrealized cast is the unsigned scalar bridge the
// frontend emits (arith.constant only carries signless integers). Reuse the
// signless carrier the intrinsic wants instead of stacking a second cast.
static Value peelUnsignedBridge(Value value)
{
    auto cast = value.getDefiningOp<UnrealizedConversionCastOp>();
    if (!cast || cast->getNumOperands() != 1)
        return value;
    Value source = cast.getOperand(0);
    auto sourceInt = dyn_cast<IntegerType>(source.getType());
    auto resultInt = dyn_cast<IntegerType>(value.getType());
    if (!sourceInt || !resultInt || sourceInt.getWidth() != resultInt.getWidth())
        return value;
    return source;
}

// SIMD atomics are GM-only scalar read-modify-writes: the architecture op
// becomes one atom intrinsic. Arch-op lowering owns the rewrite; tla-lower-ptr
// only turns !tla.ptr into i64 byte addresses.
template <typename AtomicOp, AtomicOpKind kKind>
struct LowerSimdAtomicOp : public OpRewritePattern<AtomicOp> {
    using OpRewritePattern<AtomicOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(AtomicOp op, PatternRewriter& rewriter) const override
    {
        auto ptrType = dyn_cast<::tla::PtrType>(op.getPtr().getType());
        if (!ptrType)
            return rewriter.notifyMatchFailure(op, "address is not !tla.ptr");
        if (ptrType.getAddrspace() != AddressSpace::gm)
            return rewriter.notifyMatchFailure(op, "SIMD atomics are gm-only");
        Type elementType = ptrType.getPointee();
        Type storageType = elementType;
        if (auto integer = dyn_cast<IntegerType>(elementType))
            storageType = rewriter.getIntegerType(integer.getWidth());

        Value address = getAtomicAddress(op.getPtr());
        if (!address || !address.getType().isInteger(64))
            return rewriter.notifyMatchFailure(op, "atom address is not materialized from an i64 byte address");

        Location loc = op.getLoc();
        Type pointerType =
            LLVM::LLVMPointerType::get(rewriter.getContext(), static_cast<unsigned>(hivm::AddressSpace::GM));
        Value pointer = rewriter.create<LLVM::IntToPtrOp>(loc, pointerType, address);

        Operation* bridge = op.getValue().getDefiningOp();
        Value value = peelUnsignedBridge(op.getValue());
        if (value.getType() != storageType) {
            value = rewriter.create<UnrealizedConversionCastOp>(loc, TypeRange{storageType}, ValueRange{value})
                        .getResult(0);
        }
        // 0 is L2_CACHE_HINT_NORMAL_FV, the hint the SIMT atom lowering uses.
        Value cacheHint = rewriter.create<arith::ConstantIntOp>(loc, 0, 32);
        std::string intrinsic =
            atomicIntrinsicName(kKind == AtomicOpKind::Add ? "ADD" : "EXCH", /*gm=*/true, elementType, kKind);
        if (intrinsic.empty())
            return rewriter.notifyMatchFailure(op, "no atom intrinsic for this element type");

        Value oldValue =
            rewriter.create<LLVM::CallIntrinsicOp>(loc, storageType, intrinsic, ValueRange{pointer, value, cacheHint})
                .getResult(0);
        if (oldValue.getType() != elementType) {
            oldValue = rewriter.create<UnrealizedConversionCastOp>(loc, TypeRange{elementType}, ValueRange{oldValue})
                           .getResult(0);
        }
        Operation* addressOp = op.getPtr().getDefiningOp();
        rewriter.replaceOp(op, oldValue);
        // Both were rematerialized only for this consumer; drop them when no
        // other use remains, so no typed pointer or bridge survives the pass.
        if (addressOp && addressOp->use_empty())
            rewriter.eraseOp(addressOp);
        if (bridge && bridge != addressOp && isa<UnrealizedConversionCastOp>(bridge) && bridge->use_empty())
            rewriter.eraseOp(bridge);
        return success();
    }
};

static FailureOr<Type> convertTlaBoundaryTypeToHivm(Type type)
{
    // !tla.memref boundary types were removed; there is nothing to convert here.
    // Tensor boundaries are converted by the region + finalize passes after tile_view lowering
    // has materialized the memref bridge for view-typed call operands.
    (void)type;
    return failure();
}

static void insertTlaBoundaryArgumentCasts(func::FuncOp funcOp, ArrayRef<Type> originalInputs)
{
    if (funcOp.isDeclaration() || funcOp.getBody().empty())
        return;

    Block& entry = funcOp.getBody().front();
    OpBuilder builder(funcOp.getContext());
    builder.setInsertionPointToStart(&entry);
    for (auto [arg, originalType] : llvm::zip_equal(entry.getArguments(), originalInputs)) {
        if (arg.getType() == originalType)
            continue;
        auto castOp =
            builder.create<UnrealizedConversionCastOp>(funcOp.getLoc(), TypeRange{originalType}, ValueRange{arg});
        Value castResult = castOp.getResult(0);
        arg.replaceUsesWithIf(castResult, [&](OpOperand& use) { return use.getOwner() != castOp.getOperation(); });
    }
}

static void replaceZeroOperandTlaReturns(func::FuncOp funcOp)
{
    if (funcOp.isDeclaration() || funcOp.getBody().empty())
        return;

    SmallVector<::tla::ReturnOp, 8> returns;
    funcOp.walk([&](::tla::ReturnOp op) { returns.push_back(op); });
    for (::tla::ReturnOp op : returns) {
        OpBuilder builder(op);
        builder.create<func::ReturnOp>(op.getLoc());
        op.erase();
    }
}

static FailureOr<func::FuncOp> lowerTlaFuncBoundaryToHivm(::tla::FuncOp op)
{
    auto symNameAttr = op->getAttrOfType<StringAttr>("sym_name");
    auto typeAttr = op->getAttrOfType<TypeAttr>("function_type");
    if (!symNameAttr || !typeAttr) {
        op.emitError() << "expected tla.func to have sym_name and function_type";
        return failure();
    }

    auto funcType = llvm::dyn_cast<FunctionType>(typeAttr.getValue());
    if (!funcType) {
        op.emitError() << "expected tla.func function_type to be a FunctionType";
        return failure();
    }
    if (funcType.getNumResults() != 0) {
        op.emitError() << "tla-lower-block-idx does not yet convert function results";
        return failure();
    }

    SmallVector<Type, 8> originalInputs(funcType.getInputs().begin(), funcType.getInputs().end());
    SmallVector<Type, 8> convertedInputs;
    convertedInputs.reserve(originalInputs.size());
    for (Type input : originalInputs) {
        auto converted = convertTlaBoundaryTypeToHivm(input);
        convertedInputs.push_back(succeeded(converted) ? *converted : input);
    }

    OpBuilder builder(op);
    auto convertedFuncType = FunctionType::get(op.getContext(), convertedInputs, funcType.getResults());
    auto func = builder.create<func::FuncOp>(op.getLoc(), symNameAttr.getValue(), convertedFuncType);
    for (NamedAttribute attr : op->getAttrs()) {
        StringRef name = attr.getName().getValue();
        if (name == "sym_name" || name == "function_type")
            continue;
        func->setAttr(attr.getName(), attr.getValue());
    }
    func.getBody().takeBody(op.getRegion());
    if (!func.isDeclaration() && !func.getBody().empty()) {
        Block& entry = func.getBody().front();
        for (auto [arg, type] : llvm::zip_equal(entry.getArguments(), convertedInputs)) {
            arg.setType(type);
        }
    }

    insertTlaBoundaryArgumentCasts(func, originalInputs);
    replaceZeroOperandTlaReturns(func);
    op.erase();
    return func;
}

static LogicalResult convertFuncFuncBoundaryToHivm(func::FuncOp funcOp)
{
    auto funcType = funcOp.getFunctionType();
    if (funcType.getNumResults() != 0)
        return success();

    SmallVector<Type, 8> originalInputs(funcType.getInputs().begin(), funcType.getInputs().end());
    SmallVector<Type, 8> convertedInputs;
    convertedInputs.reserve(originalInputs.size());
    bool changed = false;
    for (Type input : originalInputs) {
        auto converted = convertTlaBoundaryTypeToHivm(input);
        if (succeeded(converted)) {
            convertedInputs.push_back(*converted);
            changed = true;
        } else {
            convertedInputs.push_back(input);
        }
    }
    if (!changed)
        return success();

    funcOp.setType(FunctionType::get(funcOp.getContext(), convertedInputs, funcType.getResults()));
    if (!funcOp.isDeclaration() && !funcOp.getBody().empty()) {
        Block& entry = funcOp.getBody().front();
        for (auto [arg, type] : llvm::zip_equal(entry.getArguments(), convertedInputs)) {
            arg.setType(type);
        }
    }
    insertTlaBoundaryArgumentCasts(funcOp, originalInputs);
    return success();
}

class TlaLowerBlockIdxPass : public PassWrapper<TlaLowerBlockIdxPass, OperationPass<ModuleOp>> {
public:
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TlaLowerBlockIdxPass)

    StringRef getArgument() const override
    {
        return "tla-lower-block-idx";
    }
    StringRef getName() const override
    {
        return "TlaLowerBlockIdxPass";
    }
    StringRef getDescription() const override
    {
        return "Lower Tla function boundaries and block ops to HIVM-compatible IR.";
    }

    void getDependentDialects(DialectRegistry& registry) const override
    {
        registry.insert<
            mlir::DLTIDialect, hacc::HACCDialect, hivm::HIVMDialect, arith::ArithDialect, func::FuncDialect,
            ::mlir::memref::MemRefDialect, ::tla::TlaDialect, LLVM::LLVMDialect>();
    }

    void runOnOperation() override
    {
        ModuleOp module = getOperation();

        SmallVector<::tla::FuncOp, 8> tlaFuncs;
        module.walk([&](::tla::FuncOp op) { tlaFuncs.push_back(op); });
        for (::tla::FuncOp op : tlaFuncs) {
            if (failed(lowerTlaFuncBoundaryToHivm(op))) {
                signalPassFailure();
                return;
            }
        }

        SmallVector<func::FuncOp, 8> funcs;
        module.walk([&](func::FuncOp op) { funcs.push_back(op); });
        for (func::FuncOp op : funcs) {
            if (failed(convertFuncFuncBoundaryToHivm(op))) {
                signalPassFailure();
                return;
            }
        }

        ConversionTarget target(getContext());
        target.addLegalDialect<
            mlir::DLTIDialect, hacc::HACCDialect, hivm::HIVMDialect, arith::ArithDialect, func::FuncDialect,
            ::mlir::memref::MemRefDialect, ::tla::TlaDialect, LLVM::LLVMDialect>();
        target.markUnknownOpDynamicallyLegal([](Operation*) { return true; });
        target.addIllegalOp<
            ::tla::BlockNumOp, ::tla::BlockIdxOp, ::tla::SubBlockIdxOp, ::tla::SubBlockNumOp, ::tla::GetSystemCycleOp,
            ::tla::SimdAtomicAddOp, ::tla::SimdAtomicExchOp>();

        RewritePatternSet patterns(&getContext());
        patterns.add<
            LowerBlockNumOp, LowerBlockIdxOp, LowerSubBlockIdxOp, LowerSubBlockNumOp, LowerGetSystemCycleOp,
            LowerSimdAtomicOp<::tla::SimdAtomicAddOp, AtomicOpKind::Add>,
            LowerSimdAtomicOp<::tla::SimdAtomicExchOp, AtomicOpKind::Exch>>(&getContext());

        if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
            signalPassFailure();
        }
    }
};

} // namespace

std::unique_ptr<Pass> createTlaLowerBlockIdxPass()
{
    return std::make_unique<TlaLowerBlockIdxPass>();
}

void registerTlaLowerBlockIdxPass()
{
    PassRegistration<TlaLowerBlockIdxPass>();
}

} // namespace tla
