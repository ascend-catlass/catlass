#include "Passes/TlaTensorToMemref.h"

#include "PassesCommon.h"
#include "mlir/Dialect/Utils/ReshapeOpsUtils.h"

namespace tla {

mlir::Value castValueToI64(mlir::OpBuilder& builder, mlir::Location loc, mlir::Value value)
{
    Type type = value.getType();
    if (type.isInteger(64))
        return value;
    if (type.isIndex())
        return builder.create<arith::IndexCastOp>(loc, builder.getI64Type(), value);
    if (auto intType = dyn_cast<IntegerType>(type)) {
        if (intType.getWidth() < 64)
            return builder.create<arith::ExtSIOp>(loc, builder.getI64Type(), value);
        if (intType.getWidth() > 64)
            return builder.create<arith::TruncIOp>(loc, builder.getI64Type(), value);
    }
    return value;
}

// Whether `memref.cast` can legally carry `from` to `to`. The op's own
// predicate, so a guarded builder can never construct one the verifier rejects.
static bool isMemrefCastCompatible(mlir::Type from, mlir::Type to)
{
    mlir::Type fromTypes[] = {from};
    mlir::Type toTypes[] = {to};
    return mlir::memref::CastOp::areCastCompatible(
        TypeRange(ArrayRef<Type>(fromTypes)), TypeRange(ArrayRef<Type>(toTypes)));
}

mlir::FailureOr<mlir::Value> castMemrefToType(
    mlir::OpBuilder& builder, mlir::Location loc, mlir::Value value, mlir::MemRefType memrefType)
{
    if (value.getType() == memrefType)
        return value;
    if (!isa<MemRefType>(value.getType()))
        return failure();
    if (!isMemrefCastCompatible(value.getType(), memrefType))
        return failure();
    return builder.create<mlir::memref::CastOp>(loc, memrefType, value).getResult();
}

bool hasOnlyStagedResultUsers(mlir::Operation* op, llvm::ArrayRef<mlir::Operation*> stagedErase)
{
    if (!op || op->getNumResults() == 0)
        return false;
    for (Value result : op->getResults()) {
        for (Operation* user : result.getUsers()) {
            if (!llvm::is_contained(stagedErase, user))
                return false;
        }
    }
    return true;
}

void stageDeadTensorDescriptors(mlir::Operation* root, llvm::SmallVectorImpl<mlir::Operation*>& toErase)
{
    bool progress = true;
    while (progress) {
        progress = false;
        SmallVector<Operation*, 8> newlyDead;
        root->walk([&](Operation* op) {
            if (!llvm::isa<::tla::TensorDescOp>(op) || llvm::is_contained(toErase, op))
                return;
            if (hasOnlyStagedResultUsers(op, toErase))
                newlyDead.push_back(op);
        });
        for (Operation* op : newlyDead) {
            pushStagedErase(toErase, op);
            progress = true;
        }
    }
}

void pushStagedErase(llvm::SmallVectorImpl<mlir::Operation*>* toErase, mlir::Operation* op)
{
    if (toErase && op && !llvm::is_contained(*toErase, op))
        toErase->push_back(op);
}

void pushStagedErase(llvm::SmallVectorImpl<mlir::Operation*>& toErase, mlir::Operation* op)
{
    if (!op || llvm::is_contained(toErase, op))
        return;
    toErase.push_back(op);
}

// For a GM linear-layout tensor with a static origin_shape, fill `originDims` and the
// contiguous strides implied by that origin (row-major: trailing product; column-major:
// leading product). Returns true if filled; false for non-GM, NZFamily layout, or a dynamic
// origin. Uses the raw (rank-preserving) parse so the view mirrors the declared origin.
static bool tryGmOriginLayout(
    mlir::Type tensorTy, llvm::SmallVectorImpl<int64_t>& originDims, llvm::SmallVectorImpl<int64_t>& contigStrides)
{
    auto info = parseTensorInfo(tensorTy);
    if (failed(info) || info->originShape.empty())
        return false;
    if (info->addressSpace != ::AddressSpace::gm)
        return false;
    if (info->layoutTag != LayoutTag::RowMajor && info->layoutTag != LayoutTag::ColumnMajor)
        return false;
    if (llvm::any_of(info->originShape, [](int64_t d) { return d == ShapedType::kDynamic; }))
        return false;
    unsigned rank = info->originShape.size();
    originDims.assign(info->originShape.begin(), info->originShape.end());
    contigStrides.assign(rank, 1);
    if (info->layoutTag == LayoutTag::RowMajor) {
        int64_t acc = 1;
        for (int i = rank - 1; i >= 0; --i) {
            contigStrides[i] = acc;
            acc *= originDims[i];
        }
    } else {
        int64_t acc = 1;
        for (unsigned i = 0; i < rank; ++i) {
            contigStrides[i] = acc;
            acc *= originDims[i];
        }
    }
    return true;
}

static bool isOnChipAddressSpace(StringRef addressSpace)
{
    return addressSpace == "l1" || addressSpace == "l0a" || addressSpace == "l0b" || addressSpace == "l0c" ||
           addressSpace == "ub";
}

mlir::FailureOr<mlir::Value> materializePtrValueAsMemref(
    mlir::OpBuilder& builder, mlir::Location loc, mlir::Value ptrValue, mlir::MemRefType memrefType,
    mlir::Operation* diagnosticOp, mlir::ValueRange dynamicSizes)
{
    auto intToPtr = ptrValue.getDefiningOp<::tla::IntToPtrOp>();
    if (!intToPtr) {
        diagnosticOp->emitError() << "pointer memref materialization expects the `tla.inttoptr` "
                                     "boundary produced by `tla-lower-ptr`; got: "
                                  << ptrValue;
        return failure();
    }

    unsigned expectedDynamicSizes =
        llvm::count_if(memrefType.getShape(), [](int64_t dim) { return dim == ShapedType::kDynamic; });
    if (dynamicSizes.size() != expectedDynamicSizes) {
        diagnosticOp->emitError() << "pointer_cast materialization expected " << expectedDynamicSizes
                                  << " dynamic sizes, got " << dynamicSizes.size();
        return failure();
    }
    Value address = castValueToI64(builder, loc, intToPtr.getAddr());
    if (!address.getType().isInteger(64)) {
        diagnosticOp->emitError() << "tla.inttoptr address must lower to i64, got " << address.getType();
        return failure();
    }
    return builder.create<hivm::PointerCastOp>(loc, memrefType, address, dynamicSizes).getResult();
}

mlir::FailureOr<mlir::Value> materializeDescriptorBaseMemref(
    mlir::OpBuilder& builder, mlir::Location loc, const TensorDescriptor& desc, mlir::Operation* diagnosticOp)
{
    auto memrefType = dyn_cast<MemRefType>(desc.bridgedBaseMemrefType);
    if (!memrefType)
        return failure();

    if (isa<MemRefType>(desc.base.getType()))
        return castMemrefToType(builder, loc, desc.base, memrefType);

    if (isa<::tla::PtrType>(desc.base.getType())) {
        // GM row/column-major copies back ciface stubs that take a rank-2 memref_t
        // Materialize a rank-2 identity memref sized by the tile shape.
        if (desc.addrspace == "gm" && isLinearLayout(desc.layoutTag)) {
            auto allocType = MemRefType::get(
                {ShapedType::kDynamic, ShapedType::kDynamic}, memrefType.getElementType(), AffineMap(),
                memrefType.getMemorySpace());
            SmallVector<Value, 2> dynamicSizes = {desc.shape[0], desc.shape[1]};
            return materializePtrValueAsMemref(builder, loc, desc.base, allocType, diagnosticOp, dynamicSizes);
        }
        if (auto allocationElements = getStaticAllocationElementCount(desc.base); succeeded(allocationElements)) {
            auto allocationType = MemRefType::get(
                {*allocationElements}, memrefType.getElementType(), AffineMap(), memrefType.getMemorySpace());
            return materializePtrValueAsMemref(builder, loc, desc.base, allocationType, diagnosticOp);
        }

        // A ptr-backed on-chip memref carries an address, not recoverable allocation
        // capacity. Dynamic copy/mmad shape and layout metadata travel separately in
        // TensorDescriptor, so use zero as an explicit unknown-capacity ABI sentinel
        // instead of presenting origin-shape elements as backing storage capacity.
        if (isOnChipAddressSpace(desc.addrspace) && !memrefType.hasStaticShape()) {
            Value unknownExtent = builder.create<arith::ConstantIndexOp>(loc, 0);
            SmallVector<Value, 4> unknownExtents;
            for (int64_t dim : memrefType.getShape())
                if (dim == ShapedType::kDynamic)
                    unknownExtents.push_back(unknownExtent);
            return materializePtrValueAsMemref(builder, loc, desc.base, memrefType, diagnosticOp, unknownExtents);
        }

        SmallVector<Value, 2> dynamicSizes;
        for (auto [index, dim] : llvm::enumerate(memrefType.getShape())) {
            if (dim != ShapedType::kDynamic)
                continue;
            if (memrefType.getRank() == 1) {
                dynamicSizes.push_back(builder.create<arith::MulIOp>(loc, desc.originShape[0], desc.originShape[1]));
            } else if (index == 0) {
                dynamicSizes.push_back(isLinearLayout(desc.layoutTag) ? desc.shape[0] : desc.originShape[0]);
            } else if (index == 1) {
                dynamicSizes.push_back(isLinearLayout(desc.layoutTag) ? desc.shape[1] : desc.originShape[1]);
            } else {
                diagnosticOp->emitError() << "cannot derive dynamic pointer_cast size for memref dimension " << index;
                return failure();
            }
        }
        return materializePtrValueAsMemref(builder, loc, desc.base, memrefType, diagnosticOp, dynamicSizes);
    }

    if (diagnosticOp)
        diagnosticOp->emitError() << "expected tla.tensor_desc base to be memref or !tla.ptr";
    return failure();
}

mlir::FailureOr<mlir::Value> materializeTileMemrefFromDescriptor(
    mlir::OpBuilder& builder, mlir::Location loc, const TensorDescriptor& desc, mlir::Operation* diagnosticOp,
    llvm::DenseMap<mlir::Value, mlir::Value>& baseMemrefCache)
{
    FailureOr<Value> baseMemref =
        getOrMaterializeDescriptorBaseMemref(builder, loc, desc, diagnosticOp, baseMemrefCache);
    if (failed(baseMemref))
        return failure();
    auto baseType = dyn_cast<MemRefType>((*baseMemref).getType());
    if (!baseType) {
        if (diagnosticOp)
            diagnosticOp->emitError() << "expected descriptor base to materialize to memref type";
        return failure();
    }
    // Flattened 1D buffers are already in the runtime ABI shape expected by
    // downstream lowering; avoid fabricating subviews that downstream passes
    // erase anyway.
    if (baseType.getRank() == 1)
        return *baseMemref;
    Value one = builder.create<arith::ConstantIndexOp>(loc, 1);
    Value size0 = isLinearLayout(desc.layoutTag) ? desc.shape[0] : desc.originShape[0];
    Value size1 = isLinearLayout(desc.layoutTag) ? desc.shape[1] : desc.originShape[1];
    return builder
        .create<mlir::memref::SubViewOp>(
            loc, *baseMemref, ValueRange{desc.coord[0], desc.coord[1]}, ValueRange{size0, size1}, ValueRange{one, one})
        .getResult();
}

mlir::FailureOr<mlir::Value> getOrMaterializeDescriptorBaseMemref(
    mlir::OpBuilder& builder, mlir::Location loc, const TensorDescriptor& desc, mlir::Operation* diagnosticOp,
    llvm::DenseMap<mlir::Value, mlir::Value>& baseMemrefCache)
{
    auto memrefType = dyn_cast<MemRefType>(desc.bridgedBaseMemrefType);
    if (!memrefType)
        return failure();

    // A proven alloc capacity describes one kernel-lifetime allocation object,
    // so its descriptor is shape-independent and safe to cache by pointer SSA.
    // Otherwise an inttoptr descriptor is a consumer-local view whose dynamic
    // sizes may be defined inside the current region/loop; materialize it at the
    // caller's insertion point and do not cache it by address alone.
    bool isIntToPtr = static_cast<bool>(desc.base.getDefiningOp<::tla::IntToPtrOp>());
    bool hasStaticAllocation = succeeded(getStaticAllocationElementCount(desc.base));
    if (isIntToPtr && !hasStaticAllocation)
        return materializeDescriptorBaseMemref(builder, loc, desc, diagnosticOp);

    auto it = baseMemrefCache.find(desc.base);
    if (it != baseMemrefCache.end()) {
        if (isIntToPtr)
            return it->second;
        // Return the cached base consistently with the first materialization below,
        // which returns it un-cast. A `memref.cast` to `memrefType` is only valid when
        // the ranks match; otherwise hand back the cached view directly.
        if (auto cachedType = dyn_cast<MemRefType>(it->second.getType());
            cachedType && cachedType.getRank() == memrefType.getRank())
            return castMemrefToType(builder, loc, it->second, memrefType);
        return it->second;
    }

    // Anchor the first materialization at a point that dominates every use of
    // desc.base (right after its def, or the entry of its block for a block
    // argument) so the cached memref can be reused SSA-safely by any later
    // consumer. Fall back to the caller's insertion point if neither applies.
    OpBuilder::InsertionGuard guard(builder);
    if (Operation* def = desc.base.getDefiningOp())
        builder.setInsertionPointAfter(def);
    else if (auto blockArg = dyn_cast<BlockArgument>(desc.base))
        builder.setInsertionPointToStart(blockArg.getOwner());

    FailureOr<Value> materialized = materializeDescriptorBaseMemref(builder, loc, desc, diagnosticOp);
    if (failed(materialized))
        return failure();
    baseMemrefCache[desc.base] = *materialized;
    return *materialized;
}

// ---------------------------------------------------------------------------
// Vector tile memref materialization (TlaVectorRegionPass).
// Vector-specific policy: 256-byte lane memrefs, rank-1<->2 reshape, and the
// per-copy handoff cache. Layered on the shared decode/bridge above.
// ---------------------------------------------------------------------------

FailureOr<MemRefType> getBridgedTensorMemrefType(Value tensor)
{
    return ::tla::bridgeTlaTensorType(tensor.getType());
}

FailureOr<int64_t> getStaticNumElements(ArrayRef<int64_t> shape)
{
    int64_t numElements = 1;
    for (int64_t dim : shape) {
        if (dim <= 0 || dim == ShapedType::kDynamic)
            return failure();
        numElements *= dim;
    }
    return numElements;
}

FailureOr<int64_t> getElementByteWidth(Type elementType)
{
    if (auto intType = dyn_cast<IntegerType>(elementType)) {
        int64_t width = intType.getWidth();
        if (width <= 0 || width % 8 != 0)
            return failure();
        return width / 8;
    }
    if (auto floatType = dyn_cast<FloatType>(elementType)) {
        int64_t width = floatType.getWidth();
        if (width <= 0 || width % 8 != 0)
            return failure();
        return width / 8;
    }
    return failure();
}

FailureOr<int64_t> getVectorLaneCount(Type elementType)
{
    auto elementBytes = getElementByteWidth(elementType);
    if (failed(elementBytes) || *elementBytes <= 0)
        return failure();
    constexpr int64_t kVectorBytes = 256;
    return kVectorBytes / *elementBytes;
}

FailureOr<Value> castMemrefToExpected(PatternRewriter& rewriter, Location loc, Value value, MemRefType expectedType)
{
    if (value.getType() == expectedType)
        return value;
    auto sourceType = dyn_cast<MemRefType>(value.getType());
    if (!sourceType)
        return failure();

    auto hasSameStaticElementStorage = [](MemRefType sourceType, MemRefType expectedType) {
        return sourceType.hasStaticShape() && expectedType.hasStaticShape() &&
               sourceType.getElementType() == expectedType.getElementType() &&
               sourceType.getMemorySpace() == expectedType.getMemorySpace() &&
               sourceType.getNumElements() == expectedType.getNumElements();
    };

    // Every builder below is gated on the dialect's own legality predicate, so a
    // shape/layout this helper cannot honestly adapt reports failure() to the
    // caller instead of leaving an op that only the verifier will reject -- by
    // which point the originating tensor op is no longer in the diagnostic.
    SmallVector<ReassociationIndices> reassociation{{0, 1}};

    if (sourceType.getRank() == 1 && expectedType.getRank() == 2 &&
        hasSameStaticElementStorage(sourceType, expectedType)) {
        // Same element count is not enough: the expanded layout must be the one
        // expand_shape actually produces from this source (an expected type with,
        // say, a dynamic offset or a parent row pitch is not it).
        auto expandedType =
            mlir::memref::ExpandShapeOp::computeExpandedType(sourceType, expectedType.getShape(), reassociation);
        if (succeeded(expandedType) && *expandedType == expectedType)
            return rewriter.create<mlir::memref::ExpandShapeOp>(loc, expectedType, value, reassociation).getResult();
        // Otherwise fall through to the cast guard below.
    } else if (
        sourceType.getRank() == 2 && expectedType.getRank() == 1 &&
        hasSameStaticElementStorage(sourceType, expectedType) &&
        mlir::memref::CollapseShapeOp::isGuaranteedCollapsible(sourceType, reassociation)) {
        // A tile that keeps its parent's row pitch is not collapsible; ask the
        // dialect rather than assuming a contiguous stride-1 result.
        auto collapsedType = mlir::memref::CollapseShapeOp::computeCollapsedType(sourceType, reassociation);
        Value collapsed =
            rewriter.create<mlir::memref::CollapseShapeOp>(loc, collapsedType, value, reassociation).getResult();
        if (collapsed.getType() == expectedType)
            return collapsed;
        if (isMemrefCastCompatible(collapsed.getType(), expectedType))
            return rewriter.create<mlir::memref::CastOp>(loc, expectedType, collapsed).getResult();
        return failure();
    }

    // Anything else has to go through memref.cast.
    if (!isMemrefCastCompatible(sourceType, expectedType))
        return failure();
    return rewriter.create<mlir::memref::CastOp>(loc, expectedType, value).getResult();
}

// Cast `src` to the memref type bridged from `tensor`'s `!tla.tensor` type (the
// recurring "adapt this memref to what a tensor operand expects" idiom).
static FailureOr<Value> castToBridgedType(PatternRewriter& rewriter, Location loc, Value src, Value tensor)
{
    auto expected = getBridgedTensorMemrefType(tensor);
    if (failed(expected))
        return failure();
    return castMemrefToExpected(rewriter, loc, src, *expected);
}

// The `!tla.ptr` operand of a make_tensor / make_tensor_like tensor (or the base
// of a tla.tensor_desc when it is a ptr), or null.
// The !tla.ptr base of a tile. tla-lower-tensor-desc is the sole descriptor
// producer, so every tile here is a tla.tensor_desc; return its base when it is
// a !tla.ptr (the inttoptr boundary left by tla-lower-ptr), else null.
static Value ptrOfTensorDesc(Value tensor)
{
    if (auto descOp = tensor.getDefiningOp<::tla::TensorDescOp>())
        return llvm::isa<::tla::PtrType>(descOp.getBase().getType()) ? descOp.getBase() : Value();
    return {};
}

static bool isOnChipAddressSpace(::AddressSpace addressSpace)
{
    switch (addressSpace) {
        case ::AddressSpace::l1:
        case ::AddressSpace::l0a:
        case ::AddressSpace::l0b:
        case ::AddressSpace::l0c:
        case ::AddressSpace::ub:
            return true;
        case ::AddressSpace::generic:
        case ::AddressSpace::gm:
            return false;
    }
    return false;
}

FailureOr<MemRefType> getVectorHelperArgMemrefType(Value operand)
{
    Value ptr = ptrOfTensorDesc(operand);
    if (ptr && !ptr.getDefiningOp<::tla::IntToPtrOp>())
        return failure();

    auto bridged = getBridgedTensorMemrefType(operand);
    if (failed(bridged))
        return failure();
    if (ptr) {
        if (auto allocationElements = getStaticAllocationElementCount(ptr); succeeded(allocationElements))
            return MemRefType::get(
                {*allocationElements}, bridged->getElementType(), AffineMap(), bridged->getMemorySpace());

        SmallVector<int64_t, 4> originDims, contigStrides;
        if (tryGmOriginLayout(operand.getType(), originDims, contigStrides)) {
            auto stridedLayout = StridedLayoutAttr::get(operand.getContext(), ShapedType::kDynamic, contigStrides);
            return MemRefType::get(originDims, bridged->getElementType(), stridedLayout, bridged->getMemorySpace());
        }

        auto info = parseTensorInfo(operand.getType());
        if (failed(info))
            return failure();
        if (isOnChipAddressSpace(info->addressSpace)) {
            // `bridged` describes the captured tensor view, not the backing
            // allocation. Once allocation-capacity analysis has failed, even a
            // statically shaped tile cannot prove a static base extent. Keep the
            // helper ABI flat and mark that capacity unknown; the call site supplies
            // the explicit zero sentinel required by pointer_cast.
            return MemRefType::get(
                {ShapedType::kDynamic}, bridged->getElementType(), AffineMap(), bridged->getMemorySpace());
        }
    }

    auto viewElements = getStaticNumElements(bridged->getShape());
    if (succeeded(viewElements)) {
        if (bridged->getRank() == 1)
            return *bridged;
        return MemRefType::get({*viewElements}, bridged->getElementType(), AffineMap(), bridged->getMemorySpace());
    }

    // A memref-backed dynamic rank-1 value already carries real descriptor
    // metadata.
    if (!ptr && bridged->getRank() == 1)
        return *bridged;
    return failure();
}

FailureOr<Value> materializeBaseMemref(
    PatternRewriter& rewriter, Location loc, Value tensor, DenseMap<Value, Value>* loweredMemrefByValue)
{
    if (loweredMemrefByValue) {
        auto it = loweredMemrefByValue->find(tensor);
        if (it != loweredMemrefByValue->end() && it->second)
            return castToBridgedType(rewriter, loc, it->second, tensor);
    }

    if (auto castOp = tensor.getDefiningOp<UnrealizedConversionCastOp>()) {
        if (castOp.getNumOperands() == 1 && isa<MemRefType>(castOp.getOperand(0).getType()))
            return castToBridgedType(rewriter, loc, castOp.getOperand(0), tensor);
    }

    // A descriptor backed by a kernel-argument memref views that memref directly.
    if (auto descOp = tensor.getDefiningOp<::tla::TensorDescOp>();
        descOp && isa<MemRefType>(descOp.getBase().getType()))
        return castToBridgedType(rewriter, loc, descOp.getBase(), tensor);

    // tla.tensor_desc (the sole tile producer after tla-lower-tensor-desc): the
    // tensor views its ptr's address. The ptr is the inttoptr boundary left by
    // tla-lower-ptr (any ptr_add / tensor_ptr offset was already folded into the
    // byte address), so materialize it directly via materializePtrValueAsMemref.
    if (isa_and_nonnull<::tla::TensorDescOp>(tensor.getDefiningOp())) {
        Value ptr = ptrOfTensorDesc(tensor);
        if (!ptr || !ptr.getDefiningOp<::tla::IntToPtrOp>())
            return failure();
        auto expected = getBridgedTensorMemrefType(tensor);
        if (failed(expected))
            return failure();
        auto base = materializePtrValueAsMemref(rewriter, loc, ptr, *expected, tensor.getDefiningOp());
        if (failed(base))
            return failure();
        return castToBridgedType(rewriter, loc, *base, tensor);
    }

    if (isa<MemRefType>(tensor.getType()))
        return tensor;

    return failure();
}

// Build a rank-1, `numElements`-wide reinterpret_cast of `baseMemref` at element
// `offset` (dynamic stride-1 layout). Used by the vector helper's per-lane tiles.
Value materializeFlatReinterpretSubview(
    OpBuilder& builder, Location loc, Value baseMemref, Value offset, int64_t numElements)
{
    auto baseType = cast<MemRefType>(baseMemref.getType());
    auto layout = StridedLayoutAttr::get(builder.getContext(), ShapedType::kDynamic, ArrayRef<int64_t>{1});
    auto tileType = MemRefType::get({numElements}, baseType.getElementType(), layout, baseType.getMemorySpace());
    Value size = builder.create<arith::ConstantIndexOp>(loc, numElements);
    Value stride = builder.create<arith::ConstantIndexOp>(loc, 1);
    return builder
        .create<mlir::memref::ReinterpretCastOp>(
            loc, tileType, baseMemref, offset, ValueRange{size}, ValueRange{stride})
        .getResult();
}

// ---------------------------------------------------------------------------
// Producer-side descriptor derivation owned by tla-lower-tensor-desc: seed root
// tensor values, then walk tla.tile_view / tla.make_tensor / tla.make_tensor_like
// in pre-order and produce a TensorDescriptor for every result. Downstream passes
// read the materialized tla.tensor_desc ops instead of invoking this walk.
// ---------------------------------------------------------------------------

// Descriptor consumers only read materialized tla.tensor_desc operations.
mlir::LogicalResult collectMaterializedTensorDescriptors(
    mlir::func::FuncOp funcOp, llvm::DenseMap<mlir::Value, TensorDescriptor>& descriptorByValue)
{
    descriptorByValue.clear();
    bool collectionFailed = false;
    funcOp.walk([&](Operation* op) {
        if (auto descOp = llvm::dyn_cast<::tla::TensorDescOp>(op)) {
            auto desc = descriptorFromTensorDescOp(descOp);
            if (failed(desc)) {
                collectionFailed = true;
                return;
            }
            descriptorByValue[descOp.getResult()] = *desc;
            return;
        }

        if (llvm::isa<::tla::TileViewOp, ::tla::MakeTensorOp, ::tla::MakeTensorLikeOp>(op)) {
            op->emitError(
                "raw tensor view producer reached a descriptor consumer; "
                "expected tla-lower-tensor-desc to materialize tla.tensor_desc");
            collectionFailed = true;
        }
    });
    return failure(collectionFailed);
}

// ---------------------------------------------------------------------------
// Copy-route runtime lowering (shared by tla-cube-region / tla-vector-region).
// ---------------------------------------------------------------------------

static mlir::FailureOr<hivm::AddressSpace> resolveHivmAddressSpace(MLIRContext* ctx, StringRef addressSpace)
{
    auto tlaAddressSpace = symbolizeAddressSpace(addressSpace);
    if (!tlaAddressSpace)
        return failure();
    FailureOr<Attribute> memorySpaceOr = mapTlaAddressSpaceToHivmMemspace(ctx, *tlaAddressSpace);
    if (failed(memorySpaceOr))
        return failure();
    auto memorySpaceAttr = dyn_cast<hivm::AddressSpaceAttr>(*memorySpaceOr);
    if (!memorySpaceAttr)
        return failure();
    return memorySpaceAttr.getAddressSpace();
}

static StringRef copyRuntimeElemSuffix(Type elementType)
{
    if (elementType.isF32())
        return "float";
    if (elementType.isF16())
        return "half";
    if (elementType.isBF16())
        return "bf16";
    if (elementType.isSignlessInteger(8))
        return "int8_t";
    if (elementType.isSignlessInteger(16))
        return "int16_t";
    if (elementType.isSignlessInteger(32))
        return "int32_t";
    if (elementType.isFloat8E4M3FN())
        return "fp8_e4m3fn_t";
    if (elementType.isFloat8E5M2())
        return "fp8_e5m2_t";
    return {};
}

// Vector staging shares copyRuntimeElemSuffix with the other copy routes.
// fp8 is registered on GM<->UB and UB->L1 like int8_t; fixpipe remains
// independently restricted by isLegalFixpipeElementType below.

// Whether fixpipe can carry this L0C element type out to the given destination
// type. An fp32 accumulator (the float MMAD routes) may be narrowed to f16/bf16
// on the way out, or land unchanged as f32; an i32 accumulator (the int8 MMAD
// route) has no narrowing path and must land as i32.
//
// Both sides are enumerated deliberately. Accepting any destination for an f32
// accumulator would be safe only for as long as copyRuntimeElemSuffix had no
// mapping for the other types: the route would resolve to an empty suffix and be
// rejected further down. Once a new element type gains a suffix -- as fp8 does --
// that accidental guard disappears and the route instead resolves to a callee
// name with no REGISTER_L0C_TO_* behind it, which fails at link time rather than
// as a diagnostic here.
static bool isLegalFixpipeElementType(Type srcElementType, Type dstElementType)
{
    if (srcElementType.isF32())
        return dstElementType.isF32() || dstElementType.isF16() || dstElementType.isBF16();
    return srcElementType.isSignlessInteger(32) && dstElementType.isSignlessInteger(32);
}

std::string getCopyRouteCallee(
    MLIRContext* ctx, StringRef srcAddrspace, StringRef dstAddrspace, ::LayoutTag srcLayout, ::LayoutTag dstLayout,
    Type srcElementType, Type dstElementType, StringRef extraDesc)
{
    FailureOr<hivm::AddressSpace> srcSpace = resolveHivmAddressSpace(ctx, srcAddrspace);
    FailureOr<hivm::AddressSpace> dstSpace = resolveHivmAddressSpace(ctx, dstAddrspace);
    if (failed(srcSpace) || failed(dstSpace))
        return {};
    Type dstElem = dstElementType;

    // Copy routing is keyed by explicit (addrspace, layout-tag) pairs. Runtime
    // symbol names encode both endpoint layout tags so future layout variants can
    // be added as new explicit routes instead of overloading addrspace-only names.
    if (*srcSpace == hivm::AddressSpace::UB && *dstSpace == hivm::AddressSpace::L1 &&
        srcLayout == LayoutTag::RowMajor && dstLayout == LayoutTag::zN) {
        if (srcElementType != dstElem)
            return {};
        StringRef suffix = copyRuntimeElemSuffix(srcElementType);
        if (suffix.empty())
            return {};
        return Twine("copy_ub_RowMajor_to_l1_zN_").concat(suffix).str();
    }
    if (*srcSpace == hivm::AddressSpace::UB && *dstSpace == hivm::AddressSpace::L1 &&
        (srcLayout == LayoutTag::zN || srcLayout == LayoutTag::zNUnAlign) && dstLayout == LayoutTag::zN) {
        if (srcElementType != dstElem)
            return {};
        StringRef suffix = copyRuntimeElemSuffix(srcElementType);
        if (suffix.empty())
            return {};
        return Twine("copy_ub_zN_to_l1_zN_").concat(suffix).str();
    }
    // GM (row-major) -> UB (row-major): vector-core staging load.
    if (*srcSpace == hivm::AddressSpace::GM && *dstSpace == hivm::AddressSpace::UB &&
        srcLayout == LayoutTag::RowMajor && dstLayout == LayoutTag::RowMajor) {
        if (srcElementType != dstElem)
            return {};
        StringRef suffix = copyRuntimeElemSuffix(srcElementType);
        if (suffix.empty())
            return {};
        return Twine("copy_gm_RowMajor_to_ub_RowMajor_").concat(suffix).str();
    }
    // UB (row-major) -> GM (row-major): vector-core staging store.
    if (*srcSpace == hivm::AddressSpace::UB && *dstSpace == hivm::AddressSpace::GM &&
        srcLayout == LayoutTag::RowMajor && dstLayout == LayoutTag::RowMajor) {
        if (srcElementType != dstElem)
            return {};
        StringRef suffix = copyRuntimeElemSuffix(srcElementType);
        if (suffix.empty())
            return {};
        return Twine("copy_ub_RowMajor_to_gm_RowMajor_").concat(suffix).str();
    }
    // Packed fp4 GM -> L1. The source is a plain byte buffer -- the host has no
    // fp4 type, so GM holds packed bytes -- while the destination tile is
    // logically i4. This is the one route where src and dst element types
    // legitimately differ: it is where packed bytes become fp4 elements. The
    // encoding comes from extraDesc, which the copy lowering fills from the
    // destination tile's own element type.
    // The element types are part of the *condition*, not a check inside the body:
    // fp4 now shares zN / nZ with every other operand type, so the layouts alone
    // no longer identify this route and a plain i8 zN copy has to fall through to
    // the generic one below.
    if (*srcSpace == hivm::AddressSpace::GM && *dstSpace == hivm::AddressSpace::L1 && srcElementType == dstElem &&
        ::tla::isPackedFp4Type(dstElem) &&
        ((srcLayout == LayoutTag::RowMajor && dstLayout == LayoutTag::zN) ||
         (srcLayout == LayoutTag::ColumnMajor && dstLayout == LayoutTag::nZ))) {
        // The suffix is the C++ type the wrapper instantiates, like every other
        // bc symbol. The encoding comes off the destination tile's element type;
        // there is nowhere else it could come from, and nothing to disagree with.
        StringRef fmt =
            ::llvm::isa<::tla::Float4E1M2Type>(dstElem) ? StringRef("float4_e1m2x2_t") : StringRef("float4_e2m1x2_t");
        StringRef src = stringifyLayoutTag(srcLayout);
        StringRef dst = stringifyLayoutTag(dstLayout);
        return (Twine("copy_gm_") + src + "_to_l1_" + dst + "_" + fmt).str();
    }
    // An e8m0 scale block, on either side of the copy. The dialect type is the
    // spelling that says what the bytes mean, and both GM and L1 tiles normally
    // carry it; plain i8 stays accepted because the host cannot produce an e8m0
    // buffer and a caller may hand the bytes over untyped.
    auto isScaleElem = [](Type elem) { return ::llvm::isa<::tla::Float8E8M0Type>(elem) || elem.isSignlessInteger(8); };

    // MX scale GM -> L1, with the fractal reorder done by the copy. The source
    // tag says how the block sits in GM; A-side scales land on L1 as zZMxScale and
    // B-side as nNMxScale whichever orientation they came in.
    if (*srcSpace == hivm::AddressSpace::GM && *dstSpace == hivm::AddressSpace::L1 && isScaleElem(srcElementType) &&
        isScaleElem(dstElem)) {
        StringRef src, dst;
        if (dstLayout == LayoutTag::zZMxScale &&
            (srcLayout == LayoutTag::RowMajorMxScaleA || srcLayout == LayoutTag::ColMajorMxScaleA)) {
            src = stringifyLayoutTag(srcLayout);
            dst = stringifyLayoutTag(dstLayout);
        } else if (
            dstLayout == LayoutTag::nNMxScale &&
            (srcLayout == LayoutTag::RowMajorMxScaleB || srcLayout == LayoutTag::ColMajorMxScaleB)) {
            src = stringifyLayoutTag(srcLayout);
            dst = stringifyLayoutTag(dstLayout);
        }
        if (!src.empty())
            return (Twine("copy_gm_") + src + "_to_l1_" + dst + "_uint8_t").str();
    }

    // Element types live in the condition, not a check inside the body -- a body
    // check would abort the whole lookup instead of falling through to the
    // routes below.
    if (*srcSpace == hivm::AddressSpace::GM && *dstSpace == hivm::AddressSpace::L1 &&
        srcLayout == LayoutTag::RowMajor && dstLayout == LayoutTag::zN) {
        if (srcElementType != dstElem)
            return {};
        StringRef suffix = copyRuntimeElemSuffix(srcElementType);
        if (suffix.empty())
            return {};
        return Twine("copy_gm_RowMajor_to_l1_zN_").concat(suffix).str();
    }
    if (*srcSpace == hivm::AddressSpace::GM && *dstSpace == hivm::AddressSpace::L1 &&
        srcLayout == LayoutTag::ColumnMajor && dstLayout == LayoutTag::nZ) {
        if (srcElementType != dstElem)
            return {};
        StringRef suffix = copyRuntimeElemSuffix(srcElementType);
        if (suffix.empty())
            return {};
        return Twine("copy_gm_ColumnMajor_to_l1_nZ_").concat(suffix).str();
    }
    if (*srcSpace == hivm::AddressSpace::L1 && *dstSpace == hivm::AddressSpace::L0A && srcLayout == LayoutTag::zN &&
        dstLayout == LayoutTag::zN) {
        if (srcElementType != dstElem)
            return {};
        StringRef suffix = copyRuntimeElemSuffix(srcElementType);
        if (suffix.empty())
            return {};
        return Twine("copy_l1_zN_to_l0a_zN_").concat(suffix).str();
    }
    if (*srcSpace == hivm::AddressSpace::L1 && *dstSpace == hivm::AddressSpace::L0A && srcLayout == LayoutTag::nZ &&
        dstLayout == LayoutTag::zN) {
        if (srcElementType != dstElem)
            return {};
        StringRef suffix = copyRuntimeElemSuffix(srcElementType);
        if (suffix.empty())
            return {};
        return Twine("copy_l1_nZ_to_l0a_zN_").concat(suffix).str();
    }
    if (*srcSpace == hivm::AddressSpace::L1 && *dstSpace == hivm::AddressSpace::L0B && srcLayout == LayoutTag::zN &&
        dstLayout == LayoutTag::nZ) {
        if (srcElementType != dstElem)
            return {};
        StringRef suffix = copyRuntimeElemSuffix(srcElementType);
        if (suffix.empty())
            return {};
        return Twine("copy_l1_zN_to_l0b_nZ_").concat(suffix).str();
    }
    if (*srcSpace == hivm::AddressSpace::L1 && *dstSpace == hivm::AddressSpace::L0B && srcLayout == LayoutTag::nZ &&
        dstLayout == LayoutTag::nZ) {
        if (srcElementType != dstElem)
            return {};
        StringRef suffix = copyRuntimeElemSuffix(srcElementType);
        if (suffix.empty())
            return {};
        return Twine("copy_l1_nZ_to_l0b_nZ_").concat(suffix).str();
    }
    // L0C -> GM row-major: an fp32 acc may narrow to f32 / f16 / bf16 on fixpipe;
    // an i32 acc (int8 MMAD) stays i32.
    if (*srcSpace == hivm::AddressSpace::L0C && *dstSpace == hivm::AddressSpace::GM &&
        srcLayout == LayoutTag::L0Clayout && dstLayout == LayoutTag::RowMajor) {
        if (!isLegalFixpipeElementType(srcElementType, dstElem))
            return {};
        StringRef suffix = copyRuntimeElemSuffix(dstElem);
        if (suffix.empty())
            return {};
        return Twine("copy_l0c_to_gm_RowMajor_").concat(extraDesc).concat(suffix).str();
    }
    // L0C -> GM col-major: the fixpipe NZ2DN path transposes the tile on the way
    // out, so the mirrored store needs no UB staging.
    if (*srcSpace == hivm::AddressSpace::L0C && *dstSpace == hivm::AddressSpace::GM &&
        srcLayout == LayoutTag::L0Clayout && dstLayout == LayoutTag::ColumnMajor) {
        if (!isLegalFixpipeElementType(srcElementType, dstElem))
            return {};
        StringRef suffix = copyRuntimeElemSuffix(dstElem);
        if (suffix.empty())
            return {};
        return Twine("copy_l0c_to_gm_ColumnMajor_").concat(extraDesc).concat(suffix).str();
    }
    // L0C (fp32 MMAD acc) -> UB row-major: dst may be f32 / f16 / bf16 (narrowing on fixpipe).
    if (*srcSpace == hivm::AddressSpace::L0C && *dstSpace == hivm::AddressSpace::UB &&
        srcLayout == LayoutTag::L0Clayout && dstLayout == LayoutTag::RowMajor) {
        if (!isLegalFixpipeElementType(srcElementType, dstElem))
            return {};
        StringRef suffix = copyRuntimeElemSuffix(dstElem);
        if (suffix.empty())
            return {};
        return Twine("copy_l0c_to_ub_RowMajor_").concat(extraDesc).concat(suffix).str();
    }
    // L0C (fp32 MMAD acc) -> UB col-major: dst may be f32 / f16 / bf16 (narrowing on fixpipe).
    if (*srcSpace == hivm::AddressSpace::L0C && *dstSpace == hivm::AddressSpace::UB &&
        srcLayout == LayoutTag::L0Clayout && dstLayout == LayoutTag::ColumnMajor) {
        if (!isLegalFixpipeElementType(srcElementType, dstElem))
            return {};
        StringRef suffix = copyRuntimeElemSuffix(dstElem);
        if (suffix.empty())
            return {};
        return Twine("copy_l0c_to_ub_ColumnMajor_").concat(extraDesc).concat(suffix).str();
    }
    // L0C -> L1 zN: an fp32 acc may narrow to f32 / f16 / bf16 on fixpipe; an i32
    // acc (int8 MMAD) stays i32.
    if (*srcSpace == hivm::AddressSpace::L0C && *dstSpace == hivm::AddressSpace::L1 &&
        srcLayout == LayoutTag::L0Clayout && dstLayout == LayoutTag::zN) {
        if (!isLegalFixpipeElementType(srcElementType, dstElem))
            return {};
        StringRef suffix = copyRuntimeElemSuffix(dstElem);
        if (suffix.empty())
            return {};
        return Twine("copy_l0c_to_l1_zN_").concat(extraDesc).concat(suffix).str();
    }
    return {};
}

std::string getFillCallee(::LayoutTag dstLayout, Type dstElementType)
{
    // Fill covers only the packed cube operand formats, zero-only (the MX pad
    // use case); the verifier rejects everything else before lowering. fp4
    // packs two elements per byte, so the bc wrapper instantiates the fp4 x2
    // encoding over byte storage -- the same rule as the GM->L1 fp4 copy.
    StringRef suffix;
    if (::tla::isPackedFp4Type(dstElementType)) {
        suffix = ::llvm::isa<::tla::Float4E1M2Type>(dstElementType) ? StringRef("float4_e1m2x2_t") :
                                                                      StringRef("float4_e2m1x2_t");
    } else if (::mlir::isa<::mlir::Float8E4M3FNType, ::mlir::Float8E5M2Type>(dstElementType)) {
        suffix = copyRuntimeElemSuffix(dstElementType);
    }
    if (suffix.empty())
        return {};
    if (dstLayout == LayoutTag::zN)
        return (Twine("fill_l1_zN_") + suffix).str();
    if (dstLayout == LayoutTag::nZ)
        return (Twine("fill_l1_nZ_") + suffix).str();
    return {};
}

// Unified 12-field (4D) descriptor payload for every copy route. Linear
// (RowMajor/ColumnMajor) descriptors carry shape[2]=shape[3]=stride[2]=stride[3]=1
// (enforced by validateTensorDescriptor), so the same 12-field encoding serves
// both Linear and NZFamily endpoints.
SmallVector<Value, 12> buildCopyPayloadForDescriptor(OpBuilder& builder, Location loc, const TensorDescriptor& desc)
{
    return {
        castValueToI64(builder, loc, desc.shape[0]),       castValueToI64(builder, loc, desc.shape[1]),
        castValueToI64(builder, loc, desc.shape[2]),       castValueToI64(builder, loc, desc.shape[3]),
        castValueToI64(builder, loc, desc.stride[0]),      castValueToI64(builder, loc, desc.stride[1]),
        castValueToI64(builder, loc, desc.stride[2]),      castValueToI64(builder, loc, desc.stride[3]),
        castValueToI64(builder, loc, desc.coord[0]),       castValueToI64(builder, loc, desc.coord[1]),
        castValueToI64(builder, loc, desc.originShape[0]), castValueToI64(builder, loc, desc.originShape[1]),
    };
}

SmallVector<Value, 24> buildCopyPayloadForRoute(
    OpBuilder& builder, Location loc, const TensorDescriptor& srcDesc, const TensorDescriptor& dstDesc)
{
    SmallVector<Value, 24> payload;
    auto append = [&](ArrayRef<Value> values) { payload.append(values.begin(), values.end()); };
    append(buildCopyPayloadForDescriptor(builder, loc, srcDesc));
    append(buildCopyPayloadForDescriptor(builder, loc, dstDesc));
    return payload;
}

static bool isAicTemplateRuntimeCall(StringRef name)
{
    if (name == "mmad_float_float_float" || name == "mmad_half_half_float" || name == "mmad_bf16_bf16_float" ||
        name == "mmad_int8_int8_int32" || name == "tla_print_tensor_l1_f32")
        return true;
    // FP8 MMAD: one symbol per operand-format pairing (the two formats can mix).
    // The BC symbols are named after the C++ element type the wrapper
    // instantiates, so the operands read fp8_e4m3fn_t / fp8_e5m2_t.
    if (name.starts_with("mmad_fp8_e") && name.ends_with("_float"))
        return true;
    // MX mmad: same per-pairing naming, on the mx_fp8_* / float4_* operand types.
    // fp4 carries no `mx` marker because its C++ type has none -- there is no
    // non-microscaling fp4 route for it to be confused with.
    if ((name.starts_with("mmad_mx_fp8_e") || name.starts_with("mmad_float4_e")) && name.ends_with("_float"))
        return true;
    // L1 -> L0 loads that also attach the e8m0 scale block.
    if (name.starts_with("copy_mx_l1_"))
        return true;
    // fp4 operands are packed two-per-byte, so their GM -> L1 routes are named by
    // the fp4 encoding rather than by a storage type suffix.
    if (name.starts_with("copy_gm_") && (name.ends_with("_float4_e2m1x2_t") || name.ends_with("_float4_e1m2x2_t")))
        return true;
    // MX scale GM -> L1 with the reorder done by the DMA.
    if (name.starts_with("copy_gm_") && name.find("MxScale") != StringRef::npos && name.ends_with("_uint8_t"))
        return true;
    // L1 fill templates (zN/nZ destinations, cube core).
    if (name.starts_with("fill_l1_zN_") || name.starts_with("fill_l1_nZ_"))
        return true;
    if (!(name.starts_with("copy_")))
        return false;
    if (!(name.ends_with("_float") || name.ends_with("_half") || name.ends_with("_bf16") || name.ends_with("_int8_t") ||
          name.ends_with("_int32_t") || name.ends_with("_fp8_e4m3fn_t") || name.ends_with("_fp8_e5m2_t")))
        return false;
    return name.starts_with("copy_gm_RowMajor_to_l1_zN_") || name.starts_with("copy_gm_ColumnMajor_to_l1_nZ_") ||
           name.starts_with("copy_l1_zN_to_l0a_zN_") || name.starts_with("copy_l1_nZ_to_l0a_zN_") ||
           name.starts_with("copy_l1_zN_to_l0b_nZ_") || name.starts_with("copy_l1_nZ_to_l0b_nZ_") ||
           name.starts_with("copy_l0c_to_ub_RowMajor_") || name.starts_with("copy_l0c_to_gm_RowMajor_") ||
           name.starts_with("copy_l0c_to_gm_ColumnMajor_") || name.starts_with("copy_l0c_to_ub_ColumnMajor_") ||
           name.starts_with("copy_l0c_to_l1_zN_");
}

static bool isAivTemplateRuntimeCall(StringRef name)
{
    return name.starts_with("copy_ub_RowMajor_to_l1_zN_") || name.starts_with("copy_ub_zN_to_l1_zN_") ||
           name.starts_with("copy_gm_RowMajor_to_ub_RowMajor_") || name.starts_with("copy_ub_RowMajor_to_gm_RowMajor_");
}

static void annotateAicTemplateRuntimeCall(func::FuncOp func)
{
    MLIRContext* ctx = func.getContext();
    func->setAttr(hacc::stringifyEnum(hacc::HACCToLLVMIRTranslateAttr::ALWAYS_INLINE), UnitAttr::get(ctx));
    func->setAttr(hivm::TFuncCoreTypeAttr::name, hivm::TFuncCoreTypeAttr::get(ctx, hivm::TFuncCoreType::AIC));
    func->setAttr("llvm.emit_c_interface", UnitAttr::get(ctx));
}

static void annotateAivTemplateRuntimeCall(func::FuncOp func)
{
    MLIRContext* ctx = func.getContext();
    func->setAttr(hacc::stringifyEnum(hacc::HACCToLLVMIRTranslateAttr::ALWAYS_INLINE), UnitAttr::get(ctx));
    func->setAttr(hivm::TFuncCoreTypeAttr::name, hivm::TFuncCoreTypeAttr::get(ctx, hivm::TFuncCoreType::AIV));
    func->setAttr("llvm.emit_c_interface", UnitAttr::get(ctx));
}

func::FuncOp getOrCreateRuntimeCall(
    ModuleOp module, StringRef name, ArrayRef<Type> operandTypes, ArrayRef<Type> resultTypes)
{
    if (auto existing = module.lookupSymbol<func::FuncOp>(name)) {
        if (isAicTemplateRuntimeCall(name))
            annotateAicTemplateRuntimeCall(existing);
        if (isAivTemplateRuntimeCall(name))
            annotateAivTemplateRuntimeCall(existing);
        return existing;
    }
    OpBuilder builder(module.getBodyRegion());
    builder.setInsertionPointToStart(module.getBody());
    auto funcType = builder.getFunctionType(operandTypes, resultTypes);
    auto func = builder.create<func::FuncOp>(module.getLoc(), name, funcType);
    func.setPrivate();
    if (isAicTemplateRuntimeCall(name))
        annotateAicTemplateRuntimeCall(func);
    if (isAivTemplateRuntimeCall(name))
        annotateAivTemplateRuntimeCall(func);
    return func;
}

} // namespace tla
