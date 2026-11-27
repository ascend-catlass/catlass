#include "Passes/TlaTensorToMemref.h"
#include "PassesCommon.h"
#include "PassesInternal.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"

namespace tla {
namespace {

// Build the logical scalar-access view described by tla.tensor_desc. This is
// intentionally independent of the ABI memref's physical layout: the
// descriptor carries the logical offset, shape, and strides for both GM and UB
// tensors.
static FailureOr<Value> materializeScalarView(
    OpBuilder& builder, Operation* diagnosticOp, Value tensor, llvm::DenseMap<Value, Value>& baseMemrefCache)
{
    auto descOp = tensor.getDefiningOp<::tla::TensorDescOp>();
    if (!descOp)
        return diagnosticOp->emitError("scalar access expects a materialized tla.tensor_desc source"), failure();

    FailureOr<TensorDescriptor> descOr = descriptorFromTensorDescOp(descOp);
    FailureOr<ParsedTensorInfo> rawInfo = parseTensorInfo(tensor.getType());
    if (failed(descOr) || failed(rawInfo))
        return failure();
    TensorDescriptor& desc = *descOr;
    if (desc.addrspace != "gm" && desc.addrspace != "ub")
        return diagnosticOp->emitError("scalar access requires a GM or UB tensor descriptor"), failure();
    if (!isLinearLayout(desc.layoutTag))
        return diagnosticOp->emitError("scalar access supports only linear RowMajor/ColumnMajor descriptors"),
               failure();
    size_t logicalRank = rawInfo->shape.size();
    if (logicalRank != 1 && logicalRank != 2)
        return diagnosticOp->emitError("scalar access descriptor must have logical rank 1 or 2"), failure();
    if (!validateTensorDescriptor(diagnosticOp, desc, "malformed descriptor for scalar access"))
        return failure();

    FailureOr<Value> base =
        getOrMaterializeDescriptorBaseMemref(builder, diagnosticOp->getLoc(), desc, diagnosticOp, baseMemrefCache);
    if (failed(base))
        return failure();
    auto baseType = dyn_cast<MemRefType>((*base).getType());
    if (!baseType)
        return diagnosticOp->emitError("scalar descriptor base did not materialize as memref"), failure();

    Location loc = diagnosticOp->getLoc();
    auto metadata = builder.create<mlir::memref::ExtractStridedMetadataOp>(loc, *base);
    Value rowOffset = builder.createOrFold<arith::MulIOp>(loc, desc.coord[0], desc.stride[0]);
    Value colOffset = builder.createOrFold<arith::MulIOp>(loc, desc.coord[1], desc.stride[1]);
    Value logicalOffset = builder.createOrFold<arith::AddIOp>(loc, rowOffset, colOffset);
    Value storageOffset = builder.createOrFold<arith::AddIOp>(loc, metadata.getOffset(), logicalOffset);

    SmallVector<Value, 2> sizes;
    SmallVector<Value, 2> strides;
    if (logicalRank == 1) {
        sizes.push_back(desc.shape[1]);
        strides.push_back(desc.stride[1]);
    } else {
        sizes.append({desc.shape[0], desc.shape[1]});
        strides.append({desc.stride[0], desc.stride[1]});
    }

    SmallVector<int64_t, 2> dynamicShape(logicalRank, ShapedType::kDynamic);
    SmallVector<int64_t, 2> dynamicStrides(logicalRank, ShapedType::kDynamic);
    auto layout = StridedLayoutAttr::get(builder.getContext(), ShapedType::kDynamic, dynamicStrides);
    auto viewType = MemRefType::get(dynamicShape, baseType.getElementType(), layout, baseType.getMemorySpace());
    return builder
        .create<mlir::memref::ReinterpretCastOp>(loc, viewType, metadata.getBaseBuffer(), storageOffset, sizes, strides)
        .getResult();
}

// Placement: after tla-lower-tensor-desc, before vector/cube outlining. Every
// tensor-semantic scalar operation consumes a materialized descriptor; raw ABI
// memrefs are never accepted by tla.scalar_load/store.
class TlaLowerScalarAccessPass : public PassWrapper<TlaLowerScalarAccessPass, OperationPass<ModuleOp>> {
public:
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TlaLowerScalarAccessPass)

    StringRef getArgument() const override
    {
        return "tla-lower-scalar-access";
    }
    StringRef getName() const override
    {
        return "TlaLowerScalarAccessPass";
    }
    StringRef getDescription() const override
    {
        return "Lower descriptor-based GM and UB scalar tensor accesses to memref.load/store.";
    }

    void getDependentDialects(DialectRegistry& registry) const override
    {
        registry.insert<
            arith::ArithDialect, func::FuncDialect, mlir::memref::MemRefDialect, ::tla::TlaDialect,
            hivm::HIVMDialect>();
    }

    void runOnOperation() override
    {
        llvm::DenseMap<Value, Value> baseMemrefCache;
        if (failed(lowerScalarLoads(getOperation(), baseMemrefCache)) ||
            failed(lowerScalarStores(getOperation(), baseMemrefCache)))
            signalPassFailure();
    }

private:
    static LogicalResult lowerScalarLoads(ModuleOp module, llvm::DenseMap<Value, Value>& baseMemrefCache)
    {
        SmallVector<::tla::ScalarLoadOp, 16> ops;
        module.walk([&](::tla::ScalarLoadOp op) { ops.push_back(op); });
        for (::tla::ScalarLoadOp op : ops) {
            SmallVector<Value, 2> indices(op.getIndices().begin(), op.getIndices().end());
            auto rawInfo = parseTensorInfo(op.getSource().getType());
            if (failed(rawInfo) || indices.size() != rawInfo->shape.size())
                return op.emitError("scalar_load index count must match logical tensor rank"), failure();

            OpBuilder builder(op);
            FailureOr<Value> view = materializeScalarView(builder, op.getOperation(), op.getSource(), baseMemrefCache);
            if (failed(view))
                return failure();
            Value loaded = builder.create<mlir::memref::LoadOp>(op.getLoc(), *view, indices);
            if (Value storage = op.getStorage()) {
                if (loaded.getType() != storage.getType())
                    return op.emitOpError("expected an i8 storage load");
                storage.replaceAllUsesWith(loaded);
            }
            op.getResult().replaceAllUsesWith(loaded);
            op.erase();
        }
        return success();
    }

    static LogicalResult lowerScalarStores(ModuleOp module, llvm::DenseMap<Value, Value>& baseMemrefCache)
    {
        SmallVector<::tla::ScalarStoreOp, 16> ops;
        module.walk([&](::tla::ScalarStoreOp op) { ops.push_back(op); });
        for (::tla::ScalarStoreOp op : ops) {
            SmallVector<Value, 2> indices(op.getIndices().begin(), op.getIndices().end());
            auto rawInfo = parseTensorInfo(op.getDest().getType());
            if (failed(rawInfo) || indices.size() != rawInfo->shape.size())
                return op.emitError("scalar_store index count must match logical tensor rank"), failure();

            OpBuilder builder(op);
            FailureOr<Value> view = materializeScalarView(builder, op.getOperation(), op.getDest(), baseMemrefCache);
            if (failed(view))
                return failure();
            builder.create<mlir::memref::StoreOp>(op.getLoc(), op.getValue(), *view, indices);
            op.erase();
        }
        return success();
    }
};

} // namespace

std::unique_ptr<Pass> createTlaLowerScalarAccessPass()
{
    return std::make_unique<TlaLowerScalarAccessPass>();
}

void registerTlaLowerScalarAccessPass()
{
    PassRegistration<TlaLowerScalarAccessPass>();
}

} // namespace tla
