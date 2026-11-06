#include "Passes/TlaSimtUbLimits.h"
#include "Passes/TlaTensorToMemref.h"
#include "PassesCommon.h"
#include "Passes/TlaDebugPrint.h"
#include "PassesInternal.h"
#include "TlaScratchAllocation.h"

#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Func/Transforms/FuncConversions.h"
#include "mlir/Dialect/SCF/Transforms/Patterns.h"
#include "mlir/Interfaces/FunctionInterfaces.h"

namespace tla {
namespace {

// Resolve tensor_ptr while tensor descriptors still carry source provenance.
// A pointer backed by another !tla.ptr aliases that address. A pointer backed
// by a kernel-argument memref starts at the memref's aligned address. The
// resulting inttoptr is source IR for the uniform ptr conversion below; it is
// not a long-lived lowering bridge.
static FailureOr<Value> resolveTensorBacking(Value tensor)
{
    if (isa<MemRefType, ::tla::PtrType>(tensor.getType()))
        return tensor;
    if (auto desc = tensor.getDefiningOp<::tla::TensorDescOp>())
        return resolveTensorBacking(desc.getBase());
    if (auto tileView = tensor.getDefiningOp<::tla::TileViewOp>())
        return resolveTensorBacking(tileView.getSource());
    if (auto makeTensor = tensor.getDefiningOp<::tla::MakeTensorOp>())
        return makeTensor.getPtr();
    if (auto makeTensorLike = tensor.getDefiningOp<::tla::MakeTensorLikeOp>())
        return makeTensorLike.getPtr();
    return failure();
}

static LogicalResult resolveTensorPtrOps(ModuleOp module)
{
    SmallVector<::tla::TensorPtrOp, 8> tensorPtrOps;
    module.walk([&](::tla::TensorPtrOp op) { tensorPtrOps.push_back(op); });
    for (::tla::TensorPtrOp op : tensorPtrOps) {
        if (!op || !op->getBlock())
            continue;

        FailureOr<Value> base = resolveTensorBacking(op.getSrc());
        if (failed(base)) {
            op.emitError() << "tensor_ptr source must resolve through tensor_desc, tile_view, or "
                              "make_tensor to a backing !tla.ptr or memref";
            return failure();
        }

        if (isa<::tla::PtrType>((*base).getType())) {
            op.getResult().replaceAllUsesWith(*base);
            // Leave dead tensor_ptr for the end-of-pass sweep — erasing inside
            // tla.vec.func can assert when residual uses linger.
            continue;
        }

        OpBuilder builder(op);
        Value addressIndex = builder.create<mlir::memref::ExtractAlignedPointerAsIndexOp>(op.getLoc(), *base);
        Value addressI64 = builder.create<arith::IndexCastOp>(op.getLoc(), builder.getI64Type(), addressIndex);
        Value pointer = builder.create<::tla::IntToPtrOp>(op.getLoc(), op.getResult().getType(), addressI64);
        op.getResult().replaceAllUsesWith(pointer);
    }
    return success();
}

static bool isPointerConsumerBoundary(::tla::IntToPtrOp op)
{
    if (op.getResult().use_empty())
        return false;
    return llvm::all_of(op.getResult().getUsers(), [](Operation* user) {
        return isa<::tla::MakeTensorOp, ::tla::MakeTensorLikeOp>(user);
    });
}

static Value materializePtrFromI64(OpBuilder& builder, ::tla::PtrType resultType, ValueRange inputs, Location loc)
{
    if (inputs.size() != 1 || !inputs.front().getType().isInteger(64))
        return {};
    return builder.create<::tla::IntToPtrOp>(loc, resultType, inputs.front());
}

/// UPDATE next_ptr stays ``!tla.ptr`` until VectorRegion binds the hardware
/// pointer. SCF needs i64, so materialize ``tla.ptrtoint`` (or the address
/// already stored in ``tla.inttoptr``).
static Value materializeI64FromPtr(OpBuilder& builder, IntegerType resultType, ValueRange inputs, Location loc)
{
    if (inputs.size() != 1 || resultType.getWidth() != 64)
        return {};
    Value ptr = inputs.front();
    if (!isa<::tla::PtrType>(ptr.getType()))
        return {};
    if (auto intToPtr = ptr.getDefiningOp<::tla::IntToPtrOp>()) {
        if (intToPtr.getAddr().getType().isInteger(64))
            return intToPtr.getAddr();
    }
    return builder.create<::tla::PtrToIntOp>(loc, resultType, ptr).getResult();
}

// Name of the module-level symbol that reserves the kernel's UB scratch.
static constexpr llvm::StringLiteral kUbScratchGlobalName = "tla_ub_scratch";

// Highest UB byte the scratch plan touches. This is the number the backend must
// see reserved: everything above it belongs to the compiler reserve and, for
// SIMT, to the Data Cache.
static uint64_t ubHighWaterBytes(const TlaScratchAllocationPlan& plan)
{
    uint64_t high = 0;
    for (const TlaScratchAllocation& allocation : plan.allocations) {
        if (allocation.addressSpace != ::AddressSpace::ub)
            continue;
        high = std::max(high, allocation.end);
    }
    return high;
}

// Reserve the kernel's UB as a symbol the toolchain can account for.
//
// Emitting the addresses as bare constants leaves nothing for the backend to
// count, so the object reports no statically allocated UB of its own and the
// runtime lays the SIMT Data Cache down over the kernel's own buffers -- the
// small amount that appears to work being whatever the linked Ascend C template
// happens to reserve.
//
// This is an LLVM global rather than a `memref.global` for two reasons. Only a
// symbol carrying an explicit address space survives to the object: the core
// type that the backend assigns by walking function bodies never reaches a
// module-level `memref.global`, so it is dropped from mixed kernels. And the
// linkage must be private -- an externally visible global with no initializer
// is only a declaration, so the linker reserves nothing for it.
static LLVM::GlobalOp createUbScratchGlobal(ModuleOp module, uint64_t bytes)
{
    MLIRContext* context = module.getContext();
    if (bytes == 0)
        return {};

    OpBuilder builder(context);
    builder.setInsertionPointToStart(module.getBody());
    auto arrayType = LLVM::LLVMArrayType::get(IntegerType::get(context, 8), bytes);
    return builder.create<LLVM::GlobalOp>(
        module.getLoc(), arrayType, /*isConstant=*/false, LLVM::Linkage::Private, kUbScratchGlobalName,
        /*value=*/Attribute(), /*alignment=*/0,
        /*addrSpace=*/static_cast<unsigned>(hivm::AddressSpace::UB));
}

// Where the launch-sized region actually starts: the static high-water mark
// rounded up to the region pointer's alignment. The fallback of 1 is for IR
// whose result type is not a !tla.ptr at all -- the verifier already rejects a
// zero alignment, so the rounding below can never divide by zero.
static uint64_t computeDynamicUbBaseBytes(::tla::DynamicUbBaseOp op, uint64_t ubHighWaterBytes)
{
    auto ptrType = dyn_cast<::tla::PtrType>(op.getResult().getType());
    uint64_t alignment = ptrType ? ptrType.getAlignment() : 1;
    return (ubHighWaterBytes + alignment - 1) / alignment * alignment;
}

struct LowerAllocPtrPattern : public OpConversionPattern<::tla::AllocPtrOp> {
    LowerAllocPtrPattern(
        TypeConverter& converter, MLIRContext* context, const llvm::DenseMap<Value, uint64_t>& offsetByAllocResult,
        LLVM::GlobalOp ubScratch)
        : OpConversionPattern(converter, context), offsetByAllocResult(offsetByAllocResult), ubScratch(ubScratch)
    {}

    LogicalResult matchAndRewrite(::tla::AllocPtrOp op, OpAdaptor, ConversionPatternRewriter& rewriter) const override
    {
        auto offset = offsetByAllocResult.find(op.getResult());
        if (offset == offsetByAllocResult.end() ||
            offset->second > static_cast<uint64_t>(std::numeric_limits<int64_t>::max()))
            return rewriter.notifyMatchFailure(op, "missing or overflowing static scratch offset");
        auto byteOffset = static_cast<int64_t>(offset->second);

        // Only UB is declared through a symbol: it is the one space whose
        // capacity the runtime partitions against the Data Cache, so it is the
        // one the object has to account for. L1 and L0 keep bare addresses.
        auto ptrType = dyn_cast<::tla::PtrType>(op.getResult().getType());
        bool isUb = ptrType && ptrType.getAddrspace() == ::AddressSpace::ub;

        Value address;
        if (isUb && ubScratch) {
            auto pointerType =
                LLVM::LLVMPointerType::get(op.getContext(), static_cast<unsigned>(hivm::AddressSpace::UB));
            Value scratch = rewriter.create<LLVM::AddressOfOp>(op.getLoc(), pointerType, kUbScratchGlobalName);
            Value base = rewriter.create<LLVM::PtrToIntOp>(op.getLoc(), rewriter.getI64Type(), scratch);
            if (byteOffset == 0) {
                address = base;
            } else {
                Value slot = rewriter.create<arith::ConstantIntOp>(op.getLoc(), byteOffset, 64);
                address = rewriter.create<arith::AddIOp>(op.getLoc(), base, slot);
            }
        } else {
            address = rewriter.create<arith::ConstantIntOp>(op.getLoc(), byteOffset, 64);
        }

        address.getDefiningOp()->setAttr(kAllocSizeBytesMetadataAttrName, op.getSizeBytesAttr());
        rewriter.replaceOp(op, address);
        return success();
    }

private:
    const llvm::DenseMap<Value, uint64_t>& offsetByAllocResult;
    LLVM::GlobalOp ubScratch;
};

// The region starts where the static allocations end, so its base is an
// ordinary constant. Only the extent is a launch parameter, and no code in the
// kernel needs it.
struct LowerDynamicUbBasePattern : public OpConversionPattern<::tla::DynamicUbBaseOp> {
    LowerDynamicUbBasePattern(TypeConverter& converter, MLIRContext* context, uint64_t ubHighWaterBytes)
        : OpConversionPattern(converter, context), ubHighWaterBytes(ubHighWaterBytes)
    {}

    LogicalResult matchAndRewrite(
        ::tla::DynamicUbBaseOp op, OpAdaptor, ConversionPatternRewriter& rewriter) const override
    {
        if (!isa<::tla::PtrType>(op.getResult().getType()))
            return rewriter.notifyMatchFailure(op, "result is not !tla.ptr");
        uint64_t base = computeDynamicUbBaseBytes(op, ubHighWaterBytes);
        if (base > static_cast<uint64_t>(std::numeric_limits<int64_t>::max()))
            return rewriter.notifyMatchFailure(op, "overflowing dynamic ub base");
        rewriter.replaceOpWithNewOp<arith::ConstantIntOp>(op, static_cast<int64_t>(base), 64);
        return success();
    }

private:
    uint64_t ubHighWaterBytes;
};

struct LowerIntToPtrPattern : public OpConversionPattern<::tla::IntToPtrOp> {
    using OpConversionPattern::OpConversionPattern;

    LogicalResult matchAndRewrite(
        ::tla::IntToPtrOp op, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override
    {
        Value address = castValueToI64(rewriter, op.getLoc(), adaptor.getAddr());
        if (!address.getType().isInteger(64))
            return rewriter.notifyMatchFailure(op, "address is not integer-like");
        rewriter.replaceOp(op, address);
        return success();
    }
};

struct LowerRecastPtrPattern : public OpConversionPattern<::tla::RecastPtrOp> {
    using OpConversionPattern::OpConversionPattern;

    LogicalResult matchAndRewrite(
        ::tla::RecastPtrOp op, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override
    {
        rewriter.replaceOp(op, adaptor.getSrc());
        return success();
    }
};

struct LowerPtrAddPattern : public OpConversionPattern<::tla::PtrAddOp> {
    using OpConversionPattern::OpConversionPattern;

    LogicalResult matchAndRewrite(
        ::tla::PtrAddOp op, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override
    {
        auto ptrType = dyn_cast<::tla::PtrType>(op.getPtr().getType());
        if (!ptrType)
            return rewriter.notifyMatchFailure(op, "source is not !tla.ptr");
        int64_t elementBytes = getByteSizeOfFixedWidthScalarType(ptrType.getPointee());
        if (elementBytes <= 0)
            return rewriter.notifyMatchFailure(op, "pointee has no fixed byte width");

        Value address = adaptor.getPtr();
        Value elementOffset = castValueToI64(rewriter, op.getLoc(), adaptor.getOffset());
        if (!address.getType().isInteger(64) || !elementOffset.getType().isInteger(64))
            return rewriter.notifyMatchFailure(op, "converted address/offset must be i64");

        Value byteOffset = elementOffset;
        if (elementBytes != 1) {
            Value scale = rewriter.create<arith::ConstantIntOp>(op.getLoc(), elementBytes, 64);
            byteOffset = rewriter.create<arith::MulIOp>(op.getLoc(), elementOffset, scale);
        }
        rewriter.replaceOpWithNewOp<arith::AddIOp>(op, address, byteOffset);
        return success();
    }
};

static constexpr StringRef kUbStaticBytesAttrName = "tla.ub_static_bytes";
static constexpr StringRef kUbProgrammableAttrName = "tla.ub_programmable_bytes";
// Where the launch-sized region starts, or -1 when the kernel declares none.
// It does not follow from the static byte count: the base is that count rounded
// up to the region pointer's alignment, and a kernel may have a region with no
// static allocations at all -- which is why absence needs a value outside the
// range rather than a zero base.
static constexpr StringRef kUbDynamicBaseAttrName = "tla.ub_dynamic_base";
// Set by the host before lowering: the caller took the compiler's UB reserve.
static constexpr StringRef kUbReserveReleasedAttrName = "tla.ub_reserve_released";

static bool moduleHasSimtVecFunc(ModuleOp module)
{
    bool found = false;
    module.walk([&](::tla::VecFuncOp vecFuncOp) {
        auto mode = vecFuncOp->getAttrOfType<StringAttr>("mode");
        if (mode && mode.getValue().equals_insensitive("simt")) {
            found = true;
            return WalkResult::interrupt();
        }
        return WalkResult::advance();
    });
    return found;
}

// One region per function. A second could only start above the first, whose
// extent does not exist until launch, so both lower to the same address and
// would silently alias. Carving one region into parts is the supported way to
// get several.
static LogicalResult checkSingleDynamicUbRegion(ModuleOp module)
{
    llvm::DenseMap<Operation*, ::tla::DynamicUbBaseOp> firstPerFunction;
    LogicalResult result = success();
    module.walk([&](::tla::DynamicUbBaseOp op) {
        Operation* function = op->getParentOfType<FunctionOpInterface>();
        if (!function)
            function = module;
        auto [entry, inserted] = firstPerFunction.try_emplace(function, op);
        if (inserted)
            return WalkResult::advance();
        op.emitError() << "a kernel may declare at most one dynamic UB region, but this one declares "
                          "a second. Both would start at the same address and silently overlap, "
                          "because how far the first extends is only known at launch. Declare one "
                          "region and index it at an offset, or use tla.recast_ptr to view it as "
                          "another type";
        entry->second.emitRemark() << "first dynamic UB region declared here";
        result = failure();
        return WalkResult::interrupt();
    });
    return result;
}

static LogicalResult declareUbBudget(ModuleOp module, uint64_t ubBytes)
{
    // A SIMT region anywhere puts the whole kernel on the SIMT budget: every
    // stage shares one Unified Buffer with the Data Cache.
    const bool simt = moduleHasSimtVecFunc(module);
    // The host stamps this when the caller passed both cce-disable-*-reserved-ubuf
    // switches, because bisheng options never reach a pass. Without it a kernel
    // that legitimately fills the whole buffer is refused here.
    const bool reserveReleased = module->hasAttr(kUbReserveReleasedAttrName);
    const uint64_t reserve = reserveReleased ? 0 : kUbReservedBytes;
    const uint64_t limit = ubProgrammableBytes(simt, reserveReleased);

    if (ubBytes > limit)
        return module.emitError() << "this kernel allocates " << ubBytes << " bytes of UB, past the " << limit
                                  << "-byte limit for a " << (simt ? "SIMT" : "SIMD") << " kernel. UB is "
                                  << kUbTotalBytes << " bytes, less the " << reserve << "-byte compiler reserve"
                                  << (simt ? " and at least the 32768-byte Data Cache a SIMT kernel reaches "
                                             "global memory through" :
                                             "")
                                  << ". Every UB allocation counts, including buffers only the cube or SIMD "
                                     "stages touch. Shrink them, or split the work across kernels";

    auto i64 = IntegerType::get(module.getContext(), 64);
    // Stamped for every kernel, not just SIMT ones: the host needs the
    // requirement and the limit to decide what to declare. Zero means the
    // kernel uses no static UB.
    module->setAttr(kUbStaticBytesAttrName, IntegerAttr::get(i64, ubBytes));
    module->setAttr(kUbProgrammableAttrName, IntegerAttr::get(i64, limit));

    int64_t dynamicBase = -1;
    module.walk([&](::tla::DynamicUbBaseOp op) {
        dynamicBase = std::max(dynamicBase, static_cast<int64_t>(computeDynamicUbBaseBytes(op, ubBytes)));
    });
    module->setAttr(kUbDynamicBaseAttrName, IntegerAttr::get(i64, dynamicBase));
    return success();
}

class TlaLowerPtrPass : public PassWrapper<TlaLowerPtrPass, OperationPass<ModuleOp>> {
public:
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TlaLowerPtrPass)

    StringRef getArgument() const override
    {
        return "tla-lower-ptr";
    }
    StringRef getName() const override
    {
        return "TlaLowerPtrPass";
    }
    StringRef getDescription() const override
    {
        return "Lower first-class !tla.ptr SSA values to i64 byte addresses, "
               "including structural SCF/function type conversion, and "
               "rematerialize ptr only at tensor-view consumers.";
    }
    void getDependentDialects(DialectRegistry& registry) const override
    {
        registry.insert<
            arith::ArithDialect, cf::ControlFlowDialect, func::FuncDialect, LLVM::LLVMDialect, ::tla::TlaDialect,
            mlir::memref::MemRefDialect, scf::SCFDialect>();
    }

    void runOnOperation() override
    {
        ModuleOp module = getOperation();
        MLIRContext* context = &getContext();

        FailureOr<TlaScratchAllocationPlan> allocationPlan = planTlaScratchAllocations(module);
        if (failed(allocationPlan)) {
            signalPassFailure();
            return;
        }

        if (failed(checkSingleDynamicUbRegion(module))) {
            signalPassFailure();
            return;
        }

        // Refuse an over-budget kernel before reserving anything: the error
        // names the limit and the mode, which a launch failure cannot.
        auto ubBytes = reserveVfPrintFormats(module, ubHighWaterBytes(*allocationPlan));
        if (failed(ubBytes) || failed(declareUbBudget(module, *ubBytes))) {
            signalPassFailure();
            return;
        }

        LLVM::GlobalOp ubScratch = createUbScratchGlobal(module, *ubBytes);

        // UPDATE load/store keep next-window !tla.ptr results as SSA (no
        // identity tensor_ptr placeholder). Ordinary tensor_ptr ops still resolve.
        if (failed(resolveTensorPtrOps(module))) {
            signalPassFailure();
            return;
        }

        TypeConverter converter;
        converter.addConversion([](Type type) { return type; });
        converter.addConversion([&](::tla::PtrType) -> Type { return IntegerType::get(context, 64); });
        // Structural SCF conversion remaps region arguments separately from
        // operation results. Both paths may still feed a legal tensor consumer,
        // so rematerialize the typed pointer boundary for each kind of use.
        converter.addArgumentMaterialization(materializePtrFromI64);
        converter.addSourceMaterialization(materializePtrFromI64);
        converter.addTargetMaterialization(materializeI64FromPtr);

        RewritePatternSet patterns(context);
        ConversionTarget target(*context);
        patterns.add<LowerAllocPtrPattern>(converter, context, allocationPlan->offsetByAllocResult, ubScratch);
        patterns.add<LowerDynamicUbBasePattern>(converter, context, *ubBytes);
        patterns.add<LowerIntToPtrPattern, LowerRecastPtrPattern, LowerPtrAddPattern>(converter, context);
        scf::populateSCFStructuralTypeConversionsAndLegality(converter, patterns, target);
        populateFunctionOpInterfaceTypeConversionPattern<func::FuncOp>(patterns, converter);
        populateCallOpTypeConversionPattern(patterns, converter);
        populateBranchOpInterfaceTypeConversionPattern(patterns, converter);
        populateReturnOpTypeConversionPattern(patterns, converter);

        target.addLegalOp<ModuleOp>();
        target
            .addLegalDialect<arith::ArithDialect, LLVM::LLVMDialect, mlir::memref::MemRefDialect, ::tla::TlaDialect>();
        target.addIllegalOp<::tla::AllocPtrOp, ::tla::DynamicUbBaseOp, ::tla::RecastPtrOp, ::tla::PtrAddOp>();
        target.addDynamicallyLegalOp<::tla::TensorPtrOp>([](::tla::TensorPtrOp op) { return op.use_empty(); });
        target.addDynamicallyLegalOp<::tla::IntToPtrOp>(isPointerConsumerBoundary);
        // Leave memref↔tensor scaffolding legal so tla-finalize-memref can DCE
        // dead bridge casts. Same-width integer casts are the unsigned
        // debug-print bridge: arith.constant cannot return uiN, and
        // tla-lower-debug-print peels them before LLVM.
        target.addDynamicallyLegalOp<UnrealizedConversionCastOp>([](UnrealizedConversionCastOp op) {
            if (op.getNumOperands() != 1 || op.getNumResults() != 1)
                return false;
            if (auto inTy = dyn_cast<IntegerType>(op.getOperandTypes()[0])) {
                if (auto outTy = dyn_cast<IntegerType>(op.getResultTypes()[0])) {
                    if (inTy.getWidth() == outTy.getWidth())
                        return true;
                }
            }
            auto isTensorTy = [](Type t) { return succeeded(decodeTensorTypeInfo(t)); };
            return llvm::any_of(op.getOperandTypes(), isTensorTy) || llvm::any_of(op.getResultTypes(), isTensorTy);
        });
        target.addLegalOp<::tla::MakeTensorOp, ::tla::MakeTensorLikeOp>();
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

        if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
            signalPassFailure();
            return;
        }

        // ``ptrtoint(inttoptr(i64))`` is the address itself.
        SmallVector<::tla::PtrToIntOp, 8> ptrToInts;
        module.walk([&](::tla::PtrToIntOp op) { ptrToInts.push_back(op); });
        for (::tla::PtrToIntOp op : ptrToInts) {
            if (!op || !op->getBlock())
                continue;
            auto intToPtr = op.getPtr().getDefiningOp<::tla::IntToPtrOp>();
            if (!intToPtr || !intToPtr.getAddr().getType().isInteger(64))
                continue;
            op.getResult().replaceAllUsesWith(intToPtr.getAddr());
            if (op.use_empty())
                op.erase();
        }

        SmallVector<::tla::TensorPtrOp, 8> deadTensorPtrs;
        module.walk([&](::tla::TensorPtrOp op) {
            if (op.use_empty())
                deadTensorPtrs.push_back(op);
        });
        for (::tla::TensorPtrOp op : deadTensorPtrs)
            op.erase();
    }
};

} // namespace

std::unique_ptr<Pass> createTlaLowerPtrPass()
{
    return std::make_unique<TlaLowerPtrPass>();
}

void registerTlaLowerPtrPass()
{
    PassRegistration<TlaLowerPtrPass>();
}

} // namespace tla
