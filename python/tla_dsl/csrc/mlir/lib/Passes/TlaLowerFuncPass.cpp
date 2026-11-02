#include "Passes/TlaTensorDescriptor.h"
#include "PassesCommon.h"
#include "PassesInternal.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

namespace tla {
namespace {

static bool isKernelEntryCandidate(Operation* funcOp, Region& body)
{
    if (body.empty() || isPrivateSymbol(funcOp))
        return false;
    auto kind = funcOp->getAttrOfType<hacc::HACCFuncTypeAttr>(hacc::HACCFuncTypeAttr::name);
    return !kind || kind.getFunctionKind() == hacc::HACCFuncType::DEVICE;
}

static LogicalResult validateKernelTensorArg(Operation* funcOp, unsigned argIndex, ::tla::TlaTensorType tensorType)
{
    FailureOr<ParsedTensorInfo> info = parseTensorInfo(tensorType);
    if (failed(info))
        return funcOp->emitError() << "kernel tensor argument " << argIndex << " has malformed tensor metadata";
    if (info->addressSpace != ::AddressSpace::gm)
        return funcOp->emitError() << "kernel tensor argument " << argIndex << " must use gm address space";
    if (!::tla::isLinearLayout(info->layoutTag))
        return funcOp->emitError() << "kernel tensor argument " << argIndex
                                   << " must use the RowMajor, ColumnMajor, or VectorLayout layout";

    size_t rank = info->shape.size();
    if ((rank != 1 && rank != 2) || info->strides.size() != rank || info->coord.size() != rank ||
        info->originShape.size() != rank)
        return funcOp->emitError() << "kernel tensor argument " << argIndex
                                   << " must have matching rank-1 or rank-2 shape, stride, "
                                      "coord, and origin metadata";

    if (llvm::any_of(info->coord, [](int64_t value) { return value == ShapedType::kDynamic; }))
        return funcOp->emitError() << "kernel tensor argument " << argIndex << " cannot have dynamic coordinates";
    if (!llvm::all_of(info->coord, [](int64_t value) { return value == 0; }))
        return funcOp->emitError() << "kernel tensor argument " << argIndex
                                   << " must be a root tensor with zero coordinates";
    // This C++ !tla.tensor entry path has no companion origin ABI args (unlike the
    // Python Approach A memref + origin0/1 prologue). Origin must be a concrete
    // type constant here; dynamic origin belongs on the frontend dynamic-GM path.
    if (llvm::any_of(info->originShape, [](int64_t value) { return value == ShapedType::kDynamic; }))
        return funcOp->emitError() << "kernel tensor argument " << argIndex
                                   << " cannot have dynamic origin_shape; use the frontend "
                                      "dynamic GM memref ABI for runtime origin";
    if (llvm::any_of(info->originShape, [](int64_t value) { return value <= 0; }))
        return funcOp->emitError() << "kernel tensor argument " << argIndex
                                   << " origin_shape must contain positive extents";
    if (llvm::any_of(info->shape, [](int64_t value) { return value != ShapedType::kDynamic && value <= 0; }))
        return funcOp->emitError() << "kernel tensor argument " << argIndex
                                   << " shape must contain positive or dynamic extents";
    if (llvm::any_of(info->strides, [](int64_t value) { return value != ShapedType::kDynamic && value <= 0; }))
        return funcOp->emitError() << "kernel tensor argument " << argIndex
                                   << " stride must contain positive or dynamic values";

    // Preserve padded and otherwise strided row-major ABI layouts, but reject
    // static metadata whose adjacent rows overlap. Dynamic metadata is carried
    // by the memref ABI and cannot be compared at compile time.
    if (info->layoutTag == ::LayoutTag::RowMajor && rank == 2) {
        int64_t rows = info->shape[0];
        int64_t columns = info->shape[1];
        int64_t rowStride = info->strides[0];
        int64_t columnStride = info->strides[1];
        bool isStatic = columns != ShapedType::kDynamic && rowStride != ShapedType::kDynamic &&
                        columnStride != ShapedType::kDynamic;
        if (rows != 1 && isStatic && (rowStride - 1) / columnStride < columns - 1)
            return funcOp->emitError() << "RowMajor kernel tensor argument " << argIndex
                                       << " must have non-overlapping rows: stride[0] must exceed "
                                          "(shape[1] - 1) * stride[1]";
    }

    // Column-major buffers are physically contiguous row-major buffers over the
    // transposed shape. Their logical strides are therefore recoverable only for
    // the compact column-major form.
    if (info->layoutTag == ::LayoutTag::ColumnMajor) {
        if (rank == 1) {
            if (info->strides[0] != 1)
                return funcOp->emitError()
                       << "rank-1 ColumnMajor kernel tensor argument " << argIndex << " must have compact stride 1";
        } else {
            if (info->strides[0] != 1)
                return funcOp->emitError()
                       << "ColumnMajor kernel tensor argument " << argIndex << " must have compact leading stride 1";
            int64_t rows = info->shape[0];
            int64_t columnStride = info->strides[1];
            bool recoverable =
                rows == ShapedType::kDynamic ? columnStride == ShapedType::kDynamic : columnStride == rows;
            if (!recoverable)
                return funcOp->emitError()
                       << "ColumnMajor kernel tensor argument " << argIndex << " must have compact stride [1, rows]";
        }
    }

    if (failed(bridgeTlaTensorStorageType(tensorType)))
        return funcOp->emitError() << "kernel tensor argument " << argIndex
                                   << " cannot be represented by the GM memref ABI";
    return success();
}

static LogicalResult validateFunctionTensorAbi(Operation* funcOp, FunctionType funcType, Region* body)
{
    if (llvm::any_of(funcType.getResults(), [](Type type) { return isa<::tla::TlaTensorType>(type); }))
        return funcOp->emitError("kernel tensor results are not supported by the device ABI");

    bool hasTensorArg = llvm::any_of(funcType.getInputs(), [](Type type) { return isa<::tla::TlaTensorType>(type); });
    if (!hasTensorArg)
        return success();
    if (!body || !isKernelEntryCandidate(funcOp, *body))
        return funcOp->emitError("tla.tensor function arguments are supported only on non-private device entries");

    for (auto [index, type] : llvm::enumerate(funcType.getInputs())) {
        if (auto tensorType = dyn_cast<::tla::TlaTensorType>(type))
            if (failed(validateKernelTensorArg(funcOp, index, tensorType)))
                return failure();
    }
    return success();
}

static LogicalResult validateModuleTensorAbi(ModuleOp module)
{
    for (::tla::FuncOp funcOp : module.getOps<::tla::FuncOp>()) {
        auto typeAttr = funcOp->getAttrOfType<TypeAttr>("function_type");
        auto funcType = typeAttr ? dyn_cast<FunctionType>(typeAttr.getValue()) : FunctionType();
        if (!funcType)
            return funcOp.emitError("expected function_type to be a FunctionType");
        if (failed(validateFunctionTensorAbi(funcOp, funcType, &funcOp.getBody())))
            return failure();
    }
    for (func::FuncOp funcOp : module.getOps<func::FuncOp>())
        if (failed(validateFunctionTensorAbi(funcOp, funcOp.getFunctionType(), &funcOp.getBody())))
            return failure();

    LogicalResult result = success();
    module.walk([&](func::CallOp callOp) {
        if (llvm::any_of(callOp.getOperandTypes(), [](Type type) { return isa<::tla::TlaTensorType>(type); }) ||
            llvm::any_of(callOp.getResultTypes(), [](Type type) { return isa<::tla::TlaTensorType>(type); })) {
            callOp.emitError(
                "tensor-typed func.call is not supported; tensor values must remain "
                "inside a kernel entry");
            result = failure();
        }
    });
    return result;
}

static FailureOr<Value> materializeRootTensorDescriptor(
    OpBuilder& builder, Location loc, BlockArgument base, ::tla::TlaTensorType tensorType)
{
    FailureOr<ParsedTensorInfo> rawInfo = parseTensorInfo(tensorType);
    FailureOr<TensorTypeInfo> normalizedInfo = decodeTensorTypeInfo(tensorType);
    if (failed(rawInfo) || failed(normalizedInfo))
        return failure();

    auto constant = [&](int64_t value) -> Value { return builder.create<arith::ConstantIndexOp>(loc, value); };
    SmallVector<Value, 2> shape;
    for (auto [axis, extent] : llvm::enumerate(rawInfo->shape)) {
        shape.push_back(
            extent == ShapedType::kDynamic ? builder.create<mlir::memref::DimOp>(loc, base, axis).getResult() :
                                             constant(extent));
    }

    SmallVector<Value, 2> abiStrides;
    if (::tla::isLinearLayout(rawInfo->layoutTag) && llvm::is_contained(rawInfo->strides, ShapedType::kDynamic)) {
        auto metadata = builder.create<mlir::memref::ExtractStridedMetadataOp>(loc, base);
        abiStrides.append(metadata.getStrides().begin(), metadata.getStrides().end());
    }

    SmallVector<Value, 2> stride;
    if (::tla::isLinearLayout(rawInfo->layoutTag)) {
        for (auto [axis, value] : llvm::enumerate(rawInfo->strides))
            stride.push_back(value == ShapedType::kDynamic ? abiStrides[axis] : constant(value));
    } else if (rawInfo->shape.size() == 1) {
        stride.push_back(constant(1));
    } else {
        stride.push_back(constant(1));
        stride.push_back(shape[0]);
    }

    Value zero = constant(0);
    Value one = constant(1);
    Value shape0;
    Value shape1;
    Value stride0;
    Value stride1;
    Value origin0;
    Value origin1;
    // Always materialize origin from the type's originShape constants (validated
    // non-dynamic above). Do not substitute shape extents — origin may differ.
    if (rawInfo->shape.size() == 1) {
        shape0 = constant(1);
        shape1 = shape[0];
        stride1 = stride[0];
        stride0 = builder.createOrFold<arith::MulIOp>(loc, shape1, stride1);
        origin0 = constant(1);
        origin1 = constant(rawInfo->originShape[0]);
    } else {
        shape0 = shape[0];
        shape1 = shape[1];
        stride0 = stride[0];
        stride1 = stride[1];
        origin0 = constant(rawInfo->originShape[0]);
        origin1 = constant(rawInfo->originShape[1]);
    }

    return builder
        .create<::tla::TensorDescOp>(
            loc, tensorType, base, shape0, shape1, one, one, stride0, stride1, one, one, origin0, origin1, zero, zero)
        .getResult();
}

static LogicalResult materializeKernelTensorEntryAbi(ModuleOp module)
{
    for (func::FuncOp funcOp : module.getOps<func::FuncOp>()) {
        FunctionType originalType = funcOp.getFunctionType();
        // Dynamic GM args may already be memref (frontend Approach A prologue with
        // tensor_desc). Only !tla.tensor args still need bridge + root descriptor.
        if (!llvm::any_of(originalType.getInputs(), [](Type type) { return isa<::tla::TlaTensorType>(type); }))
            continue;
        if (!hasRequiredHaccEntryAttrs(funcOp) || funcOp.isDeclaration())
            return funcOp.emitError("tensor ABI materialization requires a defined HACC device entry");

        SmallVector<Type, 8> bridgedInputs;
        bridgedInputs.reserve(originalType.getNumInputs());
        for (Type input : originalType.getInputs()) {
            if (!isa<::tla::TlaTensorType>(input)) {
                bridgedInputs.push_back(input);
                continue;
            }
            FailureOr<MemRefType> bridged = bridgeTlaTensorStorageType(cast<::tla::TlaTensorType>(input));
            if (failed(bridged))
                return funcOp.emitError("failed to bridge validated kernel tensor argument");
            bridgedInputs.push_back(*bridged);
        }

        funcOp.setType(FunctionType::get(funcOp.getContext(), bridgedInputs, originalType.getResults()));
        for (auto [arg, type] : llvm::zip_equal(funcOp.getArguments(), bridgedInputs))
            arg.setType(type);

        Block& entry = funcOp.getBody().front();
        OpBuilder builder = OpBuilder::atBlockBegin(&entry);
        for (auto [index, originalArgType] : llvm::enumerate(originalType.getInputs())) {
            auto tensorType = dyn_cast<::tla::TlaTensorType>(originalArgType);
            if (!tensorType)
                continue;
            BlockArgument arg = entry.getArgument(index);
            SmallVector<OpOperand*, 8> originalUses;
            originalUses.reserve(std::distance(arg.use_begin(), arg.use_end()));
            for (OpOperand& use : arg.getUses())
                originalUses.push_back(&use);
            builder.setInsertionPointToStart(&entry);
            FailureOr<Value> descriptor = materializeRootTensorDescriptor(builder, funcOp.getLoc(), arg, tensorType);
            if (failed(descriptor))
                return funcOp.emitError() << "failed to materialize root descriptor for argument " << index;
            auto descOp = descriptor->getDefiningOp<::tla::TensorDescOp>();
            if (!descOp)
                return funcOp.emitError() << "root descriptor for argument " << index << " is not a tla.tensor_desc";
            for (OpOperand* use : originalUses)
                use->set(*descriptor);
        }
    }
    return success();
}

// Aggregate two core types: same type stays; differing non-MIX types (or any
// MIX) promote to MIX.
hivm::TFuncCoreType promoteCoreType(std::optional<hivm::TFuncCoreType> current, hivm::TFuncCoreType observed)
{
    if (!current)
        return observed;
    if (*current == observed || *current == hivm::TFuncCoreType::MIX)
        return *current;
    return hivm::TFuncCoreType::MIX;
}

// Read and validate ``tla.core_ratio`` (DenseI64ArrayAttr of length 2).
// Returns failure when the attribute is absent.
static FailureOr<std::pair<int64_t, int64_t>> getCoreRatioAttr(Operation* op)
{
    auto attr = op->getAttrOfType<DenseI64ArrayAttr>(kCoreRatioAttrName);
    if (!attr)
        return failure();
    if (attr.size() != 2)
        return op->emitError("tla.core_ratio must be a length-2 i64 array");
    int64_t aic = attr[0];
    int64_t aiv = attr[1];
    if (!((aic == 1 && aiv == 1) || (aic == 1 && aiv == 2)))
        return op->emitError("tla.core_ratio must be [1, 1] or [1, 2]");
    return std::make_pair(aic, aiv);
}

// Ensure MIX kernels carry an explicit ``tla.core_ratio`` (default 1:2) and that
// non-MIX kernels do not.
static LogicalResult ensureCoreRatioAttr(Operation* op, hivm::TFuncCoreType coreType)
{
    FailureOr<std::pair<int64_t, int64_t>> ratio = getCoreRatioAttr(op);
    if (failed(ratio)) {
        // getCoreRatioAttr emits on malformed attrs; distinguish absent vs bad.
        if (op->hasAttr(kCoreRatioAttrName))
            return failure();
        if (coreType == hivm::TFuncCoreType::MIX) {
            op->setAttr(kCoreRatioAttrName, DenseI64ArrayAttr::get(op->getContext(), {1, 2}));
        }
        return success();
    }
    if (coreType != hivm::TFuncCoreType::MIX)
        return op->emitError("tla.core_ratio is only valid on MIX kernels");
    return success();
}

// Materialize the final HACC/HIVM entry attributes for a device function from
// its core type (hacc.entry, function_kind, hivm.func_core_type, mix_mode,
// parallel_mode, hivm.core_ratio for MIX, and the C310 regbase target). This is
// the single place that stamps the per-device-function HACC/HIVM entry metadata.
void stampFunctionHaccHivmAttrs(Operation* op, hivm::TFuncCoreType coreType)
{
    if (isPrivateSymbol(op))
        return;
    MLIRContext* ctx = op->getContext();
    setC310RegbaseTargetAttr(op, ctx);
    setRequiredHaccEntryAttrs(op, ctx);
    op->setAttr(hivm::TFuncCoreTypeAttr::name, hivm::TFuncCoreTypeAttr::get(ctx, coreType));
    StringRef mixMode = coreType == hivm::TFuncCoreType::AIV && !op->hasAttr("hivm.part_of_mix") ? "aiv" : "mix";
    op->setAttr(kMixModeAttrName, StringAttr::get(ctx, mixMode));
    op->setAttr(kParallelModeAttrName, StringAttr::get(ctx, "simd"));
    // Bridge ``tla.core_ratio`` into ``#hivm.core_ratio`` so HIVM→LLVM→hivmc can
    // emit MIX_TASK_RATION (ELF .ascend.meta type=3). Caller must have run
    // ensureCoreRatioAttr first so MIX entries carry an explicit ratio.
    if (coreType == hivm::TFuncCoreType::MIX) {
        FailureOr<std::pair<int64_t, int64_t>> ratio = getCoreRatioAttr(op);
        if (succeeded(ratio)) {
            op->setAttr(
                hivm::TCoreRatioAttr::name,
                hivm::TCoreRatioAttr::get(
                    ctx, static_cast<int32_t>(ratio->first), static_cast<int32_t>(ratio->second)));
        }
    }
}

// Lower the tla.func / tla.return containers to func.func / func.return, copying
// the (non-signature) HACC/HIVM attributes stamped above onto the new func.func.
static LogicalResult lowerTlaFuncContainers(ModuleOp module, MLIRContext* ctx)
{
    ConversionTarget target(*ctx);
    target.addLegalDialect<func::FuncDialect, ::tla::TlaDialect>();
    target.markUnknownOpDynamicallyLegal([](Operation*) { return true; });
    target.addIllegalOp<::tla::FuncOp, ::tla::ReturnOp>();

    RewritePatternSet patterns(ctx);
    patterns.add<
        LowerTlaFuncToFuncPattern<LowerTlaFuncToFuncAttrPolicy::CopyNonSignatureAttrs>,
        LowerTlaReturnToFuncReturnPattern>(ctx);
    return applyPartialConversion(module, target, std::move(patterns));
}

// Lower tla device functions to HACC func.func in a single step: classify each
// device function's core type (AIC/AIV/MIX) from the tla.cube / tla.vector
// regions it contains, stamp the per-function HACC/HIVM entry metadata, lower
// the tla.func containers to func.func, and tag the module core type + C310
// target.
//
// Region placement is mandatory (enforced by the tla op verifiers: tla.mmad and
// cube-path copies live in tla.cube; tla.vec.func, the vector compute ops, and
// GM<->UB copies live in tla.vector), so region presence alone determines the
// core type:
//
//   both tla.cube and tla.vector present -> MIX
//   tla.cube only                        -> AIC
//   tla.vector only, or no region at all  -> AIV
//
// This runs before TlaSplitMixedFuncPass, so every function still has its
// frontend regions intact here; the split fragments get their core type stamped
// by that pass directly and are not re-classified.
class TlaLowerFuncPass : public PassWrapper<TlaLowerFuncPass, OperationPass<ModuleOp>> {
public:
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TlaLowerFuncPass)

    StringRef getArgument() const override
    {
        return "tla-lower-func";
    }
    StringRef getName() const override
    {
        return "TlaLowerFuncPass";
    }
    StringRef getDescription() const override
    {
        return "Lower tla.func containers and kernel tensor ABI: classify AIC/AIV/MIX, "
               "stamp HACC/HIVM attributes, convert GM tensor arguments to memrefs, and "
               "materialize root tensor descriptors.";
    }

    void getDependentDialects(DialectRegistry& registry) const override
    {
        registry.insert<
            arith::ArithDialect, mlir::DLTIDialect, hacc::HACCDialect, hivm::HIVMDialect, func::FuncDialect,
            mlir::memref::MemRefDialect, ::tla::TlaDialect>();
    }

    // A function whose core type we should infer: skip declarations, private
    // helpers, and (once HACC-marked) host functions. Runs before HACC marking
    // too, where tla.func ops carry no function-kind attr yet.
    static bool isInferableFunc(Operation* funcOp, Region& body)
    {
        return isKernelEntryCandidate(funcOp, body);
    }

    hivm::TFuncCoreType inferFuncCoreType(Operation* funcOp)
    {
        bool hasVector = false, hasCube = false;
        funcOp->walk([&](Operation* op) {
            if (isa<::tla::VectorOp>(op))
                hasVector = true;
            else if (isa<::tla::CubeOp>(op))
                hasCube = true;
            // MIX is terminal: once both regions are seen, no later op can change
            // the decision, so stop walking.
            return (hasVector && hasCube) ? WalkResult::interrupt() : WalkResult::advance();
        });

        if (hasVector && hasCube)
            return hivm::TFuncCoreType::MIX;
        if (hasCube)
            return hivm::TFuncCoreType::AIC;
        // tla.vector only, or a region-less (empty / sync-only) function -> AIV.
        return hivm::TFuncCoreType::AIV;
    }

    void runOnOperation() override
    {
        ModuleOp module = getOperation();
        MLIRContext* ctx = &getContext();

        // Validate the complete tensor ABI surface before changing attributes,
        // containers, signatures, or block arguments.
        if (failed(validateModuleTensorAbi(module))) {
            signalPassFailure();
            return;
        }

        // 1. Classify + stamp each device function, aggregating the module core type.
        //    Done on the tla.func containers so the attrs are carried onto func.func
        //    by the CopyNonSignatureAttrs lowering below.
        std::optional<hivm::TFuncCoreType> moduleCoreType;
        auto classify = [&](Operation* funcOp, Region& body) -> LogicalResult {
            if (!isInferableFunc(funcOp, body))
                return success();
            hivm::TFuncCoreType funcCoreType = inferFuncCoreType(funcOp);
            if (failed(ensureCoreRatioAttr(funcOp, funcCoreType)))
                return failure();
            stampFunctionHaccHivmAttrs(funcOp, funcCoreType);
            moduleCoreType = promoteCoreType(moduleCoreType, funcCoreType);
            return success();
        };
        for (::tla::FuncOp funcOp : module.getOps<::tla::FuncOp>()) {
            if (failed(classify(funcOp, funcOp.getBody()))) {
                signalPassFailure();
                return;
            }
        }
        for (func::FuncOp funcOp : module.getOps<func::FuncOp>()) {
            if (failed(classify(funcOp, funcOp.getBody()))) {
                signalPassFailure();
                return;
            }
        }

        // Always tag the module core type, defaulting an empty / region-less module
        // to AIV -- the same fallback inferFuncCoreType applies per function.
        hivm::TFuncCoreType resolvedModuleCoreType = moduleCoreType.value_or(hivm::TFuncCoreType::AIV);
        module->setAttr(
            hivm::TModuleCoreTypeAttr::name,
            hivm::TModuleCoreTypeAttr::get(ctx, toModuleCoreType(resolvedModuleCoreType)));

        // 2. Lower the tla.func containers to func.func and attach the C310 module
        //    target attributes.
        if (failed(lowerTlaFuncContainers(module, ctx))) {
            signalPassFailure();
            return;
        }
        if (failed(materializeKernelTensorEntryAbi(module))) {
            signalPassFailure();
            return;
        }
        ensureC310TargetAttrs(module);
    }
};

} // namespace

std::unique_ptr<Pass> createTlaLowerFuncPass()
{
    return std::make_unique<TlaLowerFuncPass>();
}

void registerTlaLowerFuncPass()
{
    PassRegistration<TlaLowerFuncPass>();
}

} // namespace tla
