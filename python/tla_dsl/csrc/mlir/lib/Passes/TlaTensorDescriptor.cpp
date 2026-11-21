#include "Passes/TlaTensorDescriptor.h"

#include "PassesCommon.h"

#include "llvm/ADT/DenseSet.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Matchers.h"

namespace tla {

mlir::FailureOr<mlir::MemRefType> bridgeTlaTensorType(mlir::Type tlaTensorType)
{
    return bridgeTlaTensorStorageType(tlaTensorType);
}

mlir::FailureOr<TensorTypeInfo> decodeTensorTypeInfo(mlir::Type tensorType)
{
    TensorTypeInfo info;
    auto tensorTy = llvm::dyn_cast<::tla::TlaTensorType>(tensorType);
    if (!tensorTy)
        return failure();
    auto layout = tensorTy.getLayout();
    auto ptr = tensorTy.getPtr();
    if (!layout.getOrigin() || !ptr.getPointee())
        return failure();

    if (failed(::tla::getTlaIndexTreeLeaves(layout.getShape().getTree(), info.shapeDims)) ||
        failed(::tla::getTlaIndexTreeLeaves(layout.getStride().getTree(), info.strideDims)) ||
        failed(::tla::getTlaIndexTreeLeaves(tensorTy.getCoord().getTree(), info.coordDims)) ||
        failed(::tla::getTlaIndexTreeLeaves(layout.getOrigin().getTree(), info.originShapeDims)))
        return failure();

    info.elementType = ptr.getPointee();
    info.tlaAddressSpace = ptr.getAddrspace();
    info.addressSpace = stringifyAddressSpace(ptr.getAddrspace()).str();
    info.layoutTag = layout.getLayoutTag();
    if (!info.elementType || info.addressSpace.empty())
        return failure();

    info.rank = static_cast<int64_t>(info.coordDims.size());
    if (info.rank == 3) {
        const size_t expectedLeaves = isNZFamilyLayout(info.layoutTag) ? 5 : 3;
        if (info.originShapeDims.size() != 3 || info.shapeDims.size() != expectedLeaves ||
            info.strideDims.size() != expectedLeaves)
            return failure();
        info.hasBatch = true;
        info.batchDims = {
            info.shapeDims.front(), info.strideDims.front(), info.originShapeDims.front(), info.coordDims.front()};
        info.shapeDims.erase(info.shapeDims.begin());
        info.strideDims.erase(info.strideDims.begin());
        info.originShapeDims.erase(info.originShapeDims.begin());
        info.coordDims.erase(info.coordDims.begin());
        info.rank = 2;
    }
    if (isLinearLayout(info.layoutTag)) {
        if (info.rank == 1 && info.shapeDims.size() == 1 && info.strideDims.size() == 1 &&
            info.originShapeDims.size() == 1) {
            int64_t extent = info.shapeDims[0];
            int64_t stride = info.strideDims[0];
            int64_t origin = info.originShapeDims[0];
            int64_t coord = info.coordDims[0];
            info.shapeDims = {1, extent};
            info.strideDims = {extent == ShapedType::kDynamic ? ShapedType::kDynamic : stride * extent, stride};
            info.originShapeDims = {1, origin};
            info.coordDims = {0, coord};
            info.rank = 2;
        }
        if (info.rank != 2 || info.shapeDims.size() != 2 || info.strideDims.size() != 2 ||
            info.originShapeDims.size() != 2)
            return failure();
    } else if (isNZFamilyLayout(info.layoutTag)) {
        if (info.rank != 2 || info.originShapeDims.size() != 2)
            return failure();
        if (info.shapeDims.size() != 4 || info.strideDims.size() != 4)
            return failure();
    } else {
        return failure();
    }
    return info;
}

mlir::FailureOr<ParsedTensorInfo> parseTensorInfo(mlir::Type tensorType)
{
    ParsedTensorInfo info;
    if (auto tensorTy = dyn_cast<::tla::TlaTensorType>(tensorType)) {
        auto layout = tensorTy.getLayout();
        auto ptr = tensorTy.getPtr();
        if (failed(::tla::getTlaIndexTreeLeaves(layout.getShape().getTree(), info.shape)) ||
            failed(::tla::getTlaIndexTreeLeaves(layout.getOrigin().getTree(), info.originShape)) ||
            failed(::tla::getTlaIndexTreeLeaves(tensorTy.getCoord().getTree(), info.coord)) ||
            failed(::tla::getTlaIndexTreeLeaves(layout.getStride().getTree(), info.strides))) {
            return failure();
        }
        info.elementType = ptr.getPointee();
        info.addressSpace = ptr.getAddrspace();
        info.layoutTag = layout.getLayoutTag();
        return info;
    }
    return failure();
}

mlir::MemRefType getDynamicStridedMemrefType(mlir::MemRefType memrefType)
{
    SmallVector<int64_t, 4> dynamicShape(memrefType.getRank(), ShapedType::kDynamic);
    SmallVector<int64_t, 4> dynamicStrides(memrefType.getRank(), ShapedType::kDynamic);
    auto layout = StridedLayoutAttr::get(memrefType.getContext(), ShapedType::kDynamic, dynamicStrides);
    return MemRefType::get(dynamicShape, memrefType.getElementType(), layout, memrefType.getMemorySpace());
}

bool validateTensorDescriptor(mlir::Operation* op, const TensorDescriptor& desc, llvm::StringRef errorMessage)
{
    if (!desc.base || !desc.bridgedBaseMemrefType || desc.addrspace.empty() || !desc.elementType) {
        op->emitError() << errorMessage;
        return false;
    }
    for (Value value : desc.shape) {
        if (!value || !value.getType().isIndex()) {
            op->emitError() << errorMessage;
            return false;
        }
    }
    for (Value value : desc.stride) {
        if (!value || !value.getType().isIndex()) {
            op->emitError() << errorMessage;
            return false;
        }
    }
    for (Value value : desc.originShape) {
        if (!value || !value.getType().isIndex()) {
            op->emitError() << errorMessage;
            return false;
        }
    }
    for (Value value : desc.coord) {
        if (!value || !value.getType().isIndex()) {
            op->emitError() << errorMessage;
            return false;
        }
    }
    if (!desc.batch.empty()) {
        if (desc.batch.size() != 4 || llvm::any_of(desc.batch, [](Value v) { return !v || !v.getType().isIndex(); })) {
            op->emitError() << errorMessage;
            return false;
        }
    }
    if (isLinearLayout(desc.layoutTag)) {
        auto isConstantOne = [](Value value) {
            llvm::APInt constant;
            return matchPattern(value, m_ConstantInt(&constant)) && constant == 1;
        };
        if (!isConstantOne(desc.shape[2]) || !isConstantOne(desc.shape[3]) || !isConstantOne(desc.stride[2]) ||
            !isConstantOne(desc.stride[3])) {
            op->emitError() << errorMessage;
            return false;
        }
    } else if (!isNZFamilyLayout(desc.layoutTag)) {
        op->emitError() << errorMessage;
        return false;
    }
    return true;
}

// Allocation capacity is optional provenance, not part of pointer identity.
// Preserve it only across joins whose alternatives prove the same capacity;
// otherwise tensor consumers build a view from their own shape/layout.
static FailureOr<int64_t> inferStaticAllocationSizeBytes(
    Value address, llvm::DenseSet<Value> visiting,
    llvm::DenseMap<Value, int64_t> assumedCapacities = llvm::DenseMap<Value, int64_t>())
{
    if (auto assumed = assumedCapacities.find(address); assumed != assumedCapacities.end())
        return assumed->second;
    if (!visiting.insert(address).second)
        return failure();

    if (auto blockArg = dyn_cast<BlockArgument>(address)) {
        if (auto forOp = dyn_cast_or_null<scf::ForOp>(blockArg.getOwner()->getParentOp())) {
            unsigned argNumber = blockArg.getArgNumber();
            if (blockArg.getOwner() == forOp.getBody() && argNumber > 0 && argNumber - 1 < forOp.getInitArgs().size()) {
                unsigned iterArgNumber = argNumber - 1;
                auto initSize =
                    inferStaticAllocationSizeBytes(forOp.getInitArgs()[iterArgNumber], visiting, assumedCapacities);
                auto yield = dyn_cast<scf::YieldOp>(forOp.getBody()->getTerminator());
                if (failed(initSize) || !yield || iterArgNumber >= yield.getNumOperands())
                    return failure();

                // Prove the init capacity is a loop invariant. The current block
                // argument may occur recursively in the backedge expression, so use
                // the candidate capacity while checking that every yielded source
                // preserves it.
                assumedCapacities[blockArg] = *initSize;
                auto yieldSize = inferStaticAllocationSizeBytes(
                    yield.getOperand(iterArgNumber), std::move(visiting), std::move(assumedCapacities));
                if (succeeded(yieldSize) && *yieldSize == *initSize)
                    return *initSize;
            }
        }
        return failure();
    }

    Operation* def = address.getDefiningOp();
    if (!def)
        return failure();
    if (auto size = def->getAttrOfType<IntegerAttr>(kAllocSizeBytesMetadataAttrName))
        return size.getInt();
    if (auto cast = dyn_cast<UnrealizedConversionCastOp>(def)) {
        if (cast.getNumOperands() == 1)
            return inferStaticAllocationSizeBytes(
                cast.getOperand(0), std::move(visiting), std::move(assumedCapacities));
    }

    // The Python frontend represents pointer-valued joins with structured SCF.
    // Do not claim provenance support for arith.select, which tla-lower-ptr does
    // not convert.
    if (auto ifOp = dyn_cast<scf::IfOp>(def)) {
        unsigned resultNumber = cast<OpResult>(address).getResultNumber();
        scf::YieldOp thenYield = ifOp.thenYield();
        scf::YieldOp elseYield = ifOp.elseYield();
        if (!thenYield || !elseYield || resultNumber >= thenYield.getNumOperands() ||
            resultNumber >= elseYield.getNumOperands())
            return failure();
        auto thenSize = inferStaticAllocationSizeBytes(thenYield.getOperand(resultNumber), visiting, assumedCapacities);
        auto elseSize = inferStaticAllocationSizeBytes(
            elseYield.getOperand(resultNumber), std::move(visiting), std::move(assumedCapacities));
        if (succeeded(thenSize) && succeeded(elseSize) && *thenSize == *elseSize)
            return *thenSize;
        return failure();
    }
    if (auto forOp = dyn_cast<scf::ForOp>(def)) {
        unsigned resultNumber = cast<OpResult>(address).getResultNumber();
        if (resultNumber >= forOp.getInitArgs().size())
            return failure();
        auto initSize = inferStaticAllocationSizeBytes(forOp.getInitArgs()[resultNumber], visiting, assumedCapacities);
        auto yield = dyn_cast<scf::YieldOp>(forOp.getBody()->getTerminator());
        if (failed(initSize) || !yield || resultNumber >= yield.getNumOperands())
            return failure();

        assumedCapacities[forOp.getRegionIterArg(resultNumber)] = *initSize;
        auto yieldSize = inferStaticAllocationSizeBytes(
            yield.getOperand(resultNumber), std::move(visiting), std::move(assumedCapacities));
        if (succeeded(yieldSize) && *initSize == *yieldSize)
            return *initSize;
    }
    return failure();
}

FailureOr<int64_t> getStaticAllocationElementCount(Value ptr)
{
    auto ptrType = dyn_cast<::tla::PtrType>(ptr.getType());
    auto intToPtr = ptr.getDefiningOp<::tla::IntToPtrOp>();
    if (!ptrType || !intToPtr)
        return failure();
    auto sizeBytes = inferStaticAllocationSizeBytes(intToPtr.getAddr(), {});
    // Prefer storage width so Bool (IR i1, AscendC b8) yields size_bytes / 1,
    // not size_bits / 1. Fall back to logical bits for true sub-byte pointees
    // (packed fp4) that have no whole-byte storage width.
    Type pointee = ptrType.getPointee();
    int64_t elementBits = elementStorageBits(pointee);
    if (elementBits <= 0)
        elementBits = getBitSizeOfFixedWidthScalarType(pointee);
    if (failed(sizeBytes) || *sizeBytes < 0 || elementBits <= 0)
        return failure();
    int64_t sizeBits = *sizeBytes * 8;
    if (sizeBits % elementBits != 0)
        return failure();
    return sizeBits / elementBits;
}

mlir::FailureOr<TensorDescriptor> descriptorFromTensorDescOp(::tla::TensorDescOp descOp)
{
    auto info = decodeTensorTypeInfo(descOp.getResult().getType());
    if (failed(info))
        return descOp->emitError() << "tla.tensor_desc result is not a structured tla.tensor", failure();
    TensorDescriptor d;
    d.base = descOp.getBase();
    d.batch.append(descOp.getBatchInfo().begin(), descOp.getBatchInfo().end());
    if (auto memrefType = llvm::dyn_cast<MemRefType>(descOp.getBase().getType())) {
        d.bridgedBaseMemrefType = memrefType;
    } else if (llvm::isa<::tla::PtrType>(descOp.getBase().getType())) {
        // ptr-backed tile: a flat rank-1 memref of the backing element count, matching
        // the tla.make_tensor{,_like} derivation (buildHivmMemrefType). The count comes
        // from the inttoptr boundary's tla.alloc_size_bytes provenance (set by
        // tla-lower-ptr), falling back to origin_shape0 * origin_shape1.
        int64_t flatElemCount = ShapedType::kDynamic;
        if (auto n = getStaticAllocationElementCount(descOp.getBase()); succeeded(n) && *n > 0) {
            flatElemCount = *n;
        } else if (
            !info->hasBatch && info->originShapeDims.size() >= 2 && info->originShapeDims[0] != ShapedType::kDynamic &&
            info->originShapeDims[1] != ShapedType::kDynamic && info->originShapeDims[0] > 0 &&
            info->originShapeDims[1] > 0) {
            flatElemCount = info->originShapeDims[0] * info->originShapeDims[1];
        }
        auto bridged =
            buildHivmMemrefType(descOp.getContext(), {flatElemCount}, info->elementType, info->tlaAddressSpace);
        if (failed(bridged))
            return failure();
        d.bridgedBaseMemrefType = *bridged;
    } else {
        return descOp->emitError() << "base must be memref or !tla.ptr", failure();
    }
    d.shape = {descOp.getShape0(), descOp.getShape1(), descOp.getShape2(), descOp.getShape3()};
    d.stride = {descOp.getStride0(), descOp.getStride1(), descOp.getStride2(), descOp.getStride3()};
    d.originShape = {descOp.getOriginShape0(), descOp.getOriginShape1()};
    d.coord = {descOp.getCoord0(), descOp.getCoord1()};
    d.layoutTag = info->layoutTag;
    d.addrspace = info->addressSpace;
    d.elementType = info->elementType;
    return d;
}

} // namespace tla
