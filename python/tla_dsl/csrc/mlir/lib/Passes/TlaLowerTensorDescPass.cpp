#include "PassesCommon.h"
#include "PassesInternal.h"
#include "Passes/TlaTensorDescDerivation.h"

// tla-lower-tensor-desc: the single owner of tile descriptor derivation. Runs
// after tla-lower-ptr and tla-split-mixed-func, before both region passes. It
// derives descriptors for every tensor view producer
// (tla.tile_view / tla.make_tensor{,_like}), rewrites each result as one
// tla.tensor_desc, and expands tensor-valued scf.if/for/while carriers into the
// resolved base/address and descriptor index leaves. Region entries and SCF
// results are reconstructed as tla.tensor_desc. Downstream passes only collect
// and consume those materialized descriptor ops.
//
// tla.tensor_ptr is already resolved by tla-lower-ptr (which runs before this
// pass): a ptr-backed tensor_ptr folds to the underlying !tla.ptr and a
// memref-backed one to a tla.inttoptr byte address, so no tensor_ptr reaches
// here. A memref base contributes its aligned address. An inttoptr base
// contributes that integer. A raw !tla.ptr (UPDATE next window) contributes
// one ``tla.ptrtoint``; the in-loop tensor is rebuilt as inttoptr of that i64.

namespace tla {
namespace {

// Build a tla.tensor_desc from a derived descriptor, keeping the producer's tile
// type on the result. Static leaves are materialized as SSA in `desc`.
static Value buildTensorDescOp(OpBuilder& builder, Location loc, Type tensorType, const TensorDescriptor& desc)
{
    return builder
        .create<::tla::TensorDescOp>(
            loc, tensorType, desc.base, desc.shape[0], desc.shape[1], desc.shape[2], desc.shape[3], desc.stride[0],
            desc.stride[1], desc.stride[2], desc.stride[3], desc.originShape[0], desc.originShape[1], desc.coord[0],
            desc.coord[1], desc.batch)
        .getResult();
}

static bool isTensorViewProducer(Operation* op)
{
    return llvm::isa<::tla::TileViewOp, ::tla::MakeTensorOp, ::tla::MakeTensorLikeOp>(op);
}

struct CarrierExpansion {
    bool isTensor = false;
    unsigned start = 0;
    unsigned count = 1;
    Type originalType;
    TensorDescriptor prototype;
    SmallVector<Type, 13> componentTypes;
};

class TensorDescControlFlowLowering {
public:
    TensorDescControlFlowLowering(func::FuncOp funcOp, TensorDescriptorDerivation& derivation)
        : funcOp(funcOp), derivation(derivation)
    {}

    LogicalResult run()
    {
        if (failed(lowerRegion(funcOp.getBody())))
            return failure();

        WalkResult result = funcOp.walk([&](Operation* op) -> WalkResult {
            if (isTensorViewProducer(op)) {
                op->emitError("tensor view producer has no derived descriptor");
                return WalkResult::interrupt();
            }
            if (isa<scf::IfOp, scf::ForOp, scf::WhileOp>(op) &&
                llvm::any_of(op->getResultTypes(), [](Type type) { return isa<::tla::TlaTensorType>(type); })) {
                op->emitError("tensor-valued SCF carrier remains after descriptor materialization");
                return WalkResult::interrupt();
            }
            if (auto yieldOp = dyn_cast<scf::YieldOp>(op)) {
                if (llvm::any_of(
                        yieldOp.getOperandTypes(), [](Type type) { return isa<::tla::TlaTensorType>(type); })) {
                    yieldOp.emitError("tensor-valued scf.yield remains after descriptor materialization");
                    return WalkResult::interrupt();
                }
            }
            if (auto conditionOp = dyn_cast<scf::ConditionOp>(op)) {
                if (llvm::any_of(
                        conditionOp.getArgs().getTypes(), [](Type type) { return isa<::tla::TlaTensorType>(type); })) {
                    conditionOp.emitError("tensor-valued scf.condition remains after descriptor materialization");
                    return WalkResult::interrupt();
                }
            }
            return WalkResult::advance();
        });
        return failure(result.wasInterrupted());
    }

private:
    FailureOr<TensorDescriptor> lookupDescriptor(Value value, Operation* diagnosticOp)
    {
        auto it = derivation.descriptorByValue.find(value);
        if (it != derivation.descriptorByValue.end())
            return it->second;
        if (auto descOp = value.getDefiningOp<::tla::TensorDescOp>())
            return descriptorFromTensorDescOp(descOp);
        diagnosticOp->emitError("tensor-valued SCF carrier operand is not a materialized tla.tensor_desc");
        return failure();
    }

    FailureOr<std::pair<TensorDescriptor, SmallVector<Value, 13>>> getCarrierFields(
        OpBuilder& builder, Location loc, Value value, Operation* diagnosticOp)
    {
        FailureOr<TensorDescriptor> descOr = lookupDescriptor(value, diagnosticOp);
        if (failed(descOr))
            return failure();
        TensorDescriptor desc = *descOr;
        if (!validateTensorDescriptor(diagnosticOp, desc, "malformed descriptor for tensor-valued SCF carrier"))
            return failure();

        auto toI64Address = [&](Value address) -> FailureOr<Value> {
            if (address.getType().isInteger(64))
                return address;
            if (address.getType().isIndex())
                return builder.create<arith::IndexCastOp>(loc, builder.getI64Type(), address).getResult();
            diagnosticOp->emitError("tensor-valued SCF carrier address must be index or i64");
            return failure();
        };

        Value address;
        if (auto baseType = dyn_cast<MemRefType>(desc.base.getType())) {
            // A tensor type does not encode whether its storage came from a
            // kernel memref or !tla.ptr. Carry one canonical i64 address so both
            // legal provenances can meet at the same structural SCF edge.
            Value addressIndex = builder.create<mlir::memref::ExtractAlignedPointerAsIndexOp>(loc, desc.base);
            FailureOr<Value> addressOr = toI64Address(addressIndex);
            if (failed(addressOr))
                return failure();
            address = *addressOr;
            desc.bridgedBaseMemrefType = getDynamicStridedMemrefType(baseType);
        } else if (isa<::tla::PtrType>(desc.base.getType())) {
            Value addressSrc;
            if (auto intToPtr = desc.base.getDefiningOp<::tla::IntToPtrOp>())
                addressSrc = intToPtr.getAddr();
            else
                addressSrc = builder.create<::tla::PtrToIntOp>(loc, builder.getI64Type(), desc.base).getResult();
            FailureOr<Value> addressOr = toI64Address(addressSrc);
            if (failed(addressOr))
                return failure();
            address = *addressOr;
            auto bridgedType = dyn_cast<MemRefType>(desc.bridgedBaseMemrefType);
            if (!bridgedType) {
                diagnosticOp->emitError("tensor-valued SCF carrier descriptor has no bridged memref type");
                return failure();
            }
            desc.bridgedBaseMemrefType = getDynamicStridedMemrefType(bridgedType);
        } else {
            diagnosticOp->emitError("tensor-valued SCF carrier requires a resolved memref or !tla.ptr descriptor base");
            return failure();
        }

        SmallVector<Value, 13> fields{address};
        fields.append(desc.shape.begin(), desc.shape.end());
        fields.append(desc.stride.begin(), desc.stride.end());
        fields.append(desc.originShape.begin(), desc.originShape.end());
        fields.append(desc.coord.begin(), desc.coord.end());
        fields.append(desc.batch.begin(), desc.batch.end());
        return std::make_pair(std::move(desc), std::move(fields));
    }

    FailureOr<CarrierExpansion> appendExpandedValue(
        OpBuilder& builder, Location loc, Value value, SmallVectorImpl<Value>& flatValues,
        SmallVectorImpl<Type>& flatTypes, Operation* diagnosticOp)
    {
        CarrierExpansion expansion;
        expansion.start = flatValues.size();
        expansion.originalType = value.getType();
        if (!isa<::tla::TlaTensorType>(value.getType())) {
            flatValues.push_back(value);
            flatTypes.push_back(value.getType());
            return expansion;
        }

        FailureOr<std::pair<TensorDescriptor, SmallVector<Value, 13>>> carrier =
            getCarrierFields(builder, loc, value, diagnosticOp);
        if (failed(carrier))
            return failure();
        expansion.isTensor = true;
        expansion.prototype = std::move(carrier->first);
        expansion.count = carrier->second.size();
        for (Value field : carrier->second) {
            flatValues.push_back(field);
            flatTypes.push_back(field.getType());
            expansion.componentTypes.push_back(field.getType());
        }
        return expansion;
    }

    LogicalResult appendUsingExpansion(
        OpBuilder& builder, Location loc, Value value, const CarrierExpansion& expansion,
        SmallVectorImpl<Value>& flatValues, Operation* diagnosticOp)
    {
        if (!expansion.isTensor) {
            if (value.getType() != expansion.originalType) {
                diagnosticOp->emitError("non-tensor SCF carrier type changed during descriptor lowering");
                return failure();
            }
            flatValues.push_back(value);
            return success();
        }

        FailureOr<std::pair<TensorDescriptor, SmallVector<Value, 13>>> carrier =
            getCarrierFields(builder, loc, value, diagnosticOp);
        if (failed(carrier))
            return failure();
        if (carrier->second.size() != expansion.count) {
            diagnosticOp->emitError("tensor descriptor field count differs across SCF carrier edges");
            return failure();
        }
        for (auto [index, field] : llvm::enumerate(carrier->second)) {
            if (field.getType() != expansion.componentTypes[index]) {
                diagnosticOp->emitError("tensor descriptor field types differ across SCF carrier edges");
                return failure();
            }
            flatValues.push_back(field);
        }
        return success();
    }

    FailureOr<Value> buildDescFromComponents(
        OpBuilder& builder, Location loc, const CarrierExpansion& expansion, ValueRange components,
        Operation* diagnosticOp)
    {
        if (!expansion.isTensor || components.size() != expansion.count ||
            components.size() != 13 + expansion.prototype.batch.size()) {
            diagnosticOp->emitError("invalid flattened tensor descriptor SCF carrier");
            return failure();
        }

        auto tensorType = dyn_cast<::tla::TlaTensorType>(expansion.originalType);
        if (!tensorType || !components[0].getType().isInteger(64)) {
            diagnosticOp->emitError("invalid tensor type or address in flattened SCF carrier");
            return failure();
        }

        TensorDescriptor desc = expansion.prototype;
        desc.base = builder.create<::tla::IntToPtrOp>(loc, tensorType.getPtr(), components[0]);
        desc.shape = {components[1], components[2], components[3], components[4]};
        desc.stride = {components[5], components[6], components[7], components[8]};
        desc.originShape = {components[9], components[10]};
        desc.coord = {components[11], components[12]};
        desc.batch.clear();
        if (components.size() == 17)
            desc.batch.append(components.begin() + 13, components.end());
        if (isLinearLayout(desc.layoutTag)) {
            // Structural carriers retain all four slots, but the materialized
            // linear descriptor must expose literal unit padding to its verifier.
            Value one = builder.create<arith::ConstantIndexOp>(loc, 1);
            desc.shape[2] = one;
            desc.shape[3] = one;
            desc.stride[2] = one;
            desc.stride[3] = one;
        }

        return buildTensorDescOp(builder, loc, expansion.originalType, desc);
    }

    LogicalResult lowerDerivableTensorProducers()
    {
        if (failed(derivation.derive(funcOp)))
            return failure();

        SmallVector<Operation*, 16> producers;
        funcOp.walk([&](Operation* op) {
            if (isTensorViewProducer(op))
                producers.push_back(op);
        });

        for (Operation* op : producers) {
            if (!op || !op->getBlock())
                continue;
            auto it = derivation.descriptorByValue.find(op->getResult(0));
            if (it == derivation.descriptorByValue.end())
                continue;
            const TensorDescriptor& desc = it->second;
            if (!validateTensorDescriptor(op, desc, "malformed descriptor for tla.tensor_desc lowering"))
                return failure();
            OpBuilder builder(op);
            Value descValue = buildTensorDescOp(builder, op->getLoc(), op->getResult(0).getType(), desc);
            op->getResult(0).replaceAllUsesWith(descValue);
            op->erase();
        }
        return success();
    }

    LogicalResult lowerRegion(Region& region)
    {
        if (failed(lowerDerivableTensorProducers()))
            return failure();

        for (Block& block : region) {
            SmallVector<Operation*, 16> operations;
            for (Operation& op : block)
                operations.push_back(&op);

            for (Operation* op : operations) {
                if (!op->getBlock())
                    continue;
                if (auto ifOp = dyn_cast<scf::IfOp>(op)) {
                    if (llvm::any_of(
                            ifOp.getResultTypes(), [](Type type) { return isa<::tla::TlaTensorType>(type); })) {
                        if (failed(lowerIf(ifOp)))
                            return failure();
                    } else {
                        if (failed(lowerRegion(ifOp.getThenRegion())) || failed(lowerRegion(ifOp.getElseRegion())))
                            return failure();
                    }
                    continue;
                }
                if (auto forOp = dyn_cast<scf::ForOp>(op)) {
                    if (llvm::any_of(
                            forOp.getResultTypes(), [](Type type) { return isa<::tla::TlaTensorType>(type); })) {
                        if (failed(lowerFor(forOp)))
                            return failure();
                    } else if (failed(lowerRegion(forOp.getRegion()))) {
                        return failure();
                    }
                    continue;
                }
                if (auto whileOp = dyn_cast<scf::WhileOp>(op)) {
                    if (llvm::any_of(
                            whileOp.getResultTypes(), [](Type type) { return isa<::tla::TlaTensorType>(type); })) {
                        if (failed(lowerWhile(whileOp)))
                            return failure();
                    } else {
                        if (failed(lowerRegion(whileOp.getBefore())) || failed(lowerRegion(whileOp.getAfter())))
                            return failure();
                    }
                    continue;
                }
                for (Region& nested : op->getRegions())
                    if (failed(lowerRegion(nested)))
                        return failure();
            }
        }
        return lowerDerivableTensorProducers();
    }

    LogicalResult lowerIf(scf::IfOp ifOp)
    {
        if (failed(lowerRegion(ifOp.getThenRegion())) || failed(lowerRegion(ifOp.getElseRegion())))
            return failure();

        scf::YieldOp thenYield = ifOp.thenYield();
        scf::YieldOp elseYield = ifOp.elseYield();
        if (!thenYield || !elseYield || thenYield.getNumOperands() != ifOp.getNumResults() ||
            elseYield.getNumOperands() != ifOp.getNumResults()) {
            ifOp.emitError("tensor-valued scf.if requires matching then/else yields");
            return failure();
        }

        SmallVector<Value, 32> thenValues;
        SmallVector<Value, 32> elseValues;
        SmallVector<Type, 32> resultTypes;
        SmallVector<CarrierExpansion, 4> expansions;
        OpBuilder thenBuilder(thenYield);
        OpBuilder elseBuilder(elseYield);
        for (unsigned i = 0, e = ifOp.getNumResults(); i < e; ++i) {
            FailureOr<CarrierExpansion> expansion = appendExpandedValue(
                thenBuilder, thenYield.getLoc(), thenYield.getOperand(i), thenValues, resultTypes, ifOp);
            if (failed(expansion) ||
                failed(appendUsingExpansion(
                    elseBuilder, elseYield.getLoc(), elseYield.getOperand(i), *expansion, elseValues, ifOp)))
                return failure();
            expansions.push_back(std::move(*expansion));
        }

        OpBuilder builder(ifOp);
        auto newIf = builder.create<scf::IfOp>(
            ifOp.getLoc(), resultTypes, ifOp.getCondition(),
            /*addThenBlock=*/false, /*addElseBlock=*/false);
        newIf->setAttrs(ifOp->getAttrs());
        newIf.getThenRegion().takeBody(ifOp.getThenRegion());
        newIf.getElseRegion().takeBody(ifOp.getElseRegion());

        OpBuilder newThenBuilder(thenYield);
        newThenBuilder.create<scf::YieldOp>(thenYield.getLoc(), thenValues);
        thenYield.erase();
        OpBuilder newElseBuilder(elseYield);
        newElseBuilder.create<scf::YieldOp>(elseYield.getLoc(), elseValues);
        elseYield.erase();

        OpBuilder resultBuilder(newIf);
        resultBuilder.setInsertionPointAfter(newIf);
        ValueRange flatResults = newIf.getResults();
        SmallVector<Value, 4> replacements;
        for (const CarrierExpansion& expansion : expansions) {
            if (!expansion.isTensor) {
                replacements.push_back(flatResults[expansion.start]);
                continue;
            }
            FailureOr<Value> desc = buildDescFromComponents(
                resultBuilder, ifOp.getLoc(), expansion, flatResults.slice(expansion.start, expansion.count), ifOp);
            if (failed(desc))
                return failure();
            replacements.push_back(*desc);
        }

        for (auto [oldResult, replacement] : llvm::zip(ifOp.getResults(), replacements))
            oldResult.replaceAllUsesWith(replacement);
        ifOp.erase();
        return success();
    }

    LogicalResult lowerFor(scf::ForOp forOp)
    {
        if (failed(lowerDerivableTensorProducers()))
            return failure();
        if (forOp.getInitArgs().size() != forOp.getNumResults()) {
            forOp.emitError("tensor-valued scf.for has inconsistent init/result counts");
            return failure();
        }

        SmallVector<Value, 32> initValues;
        SmallVector<Type, 32> resultTypes;
        SmallVector<CarrierExpansion, 4> expansions;
        OpBuilder builder(forOp);
        for (Value init : forOp.getInitArgs()) {
            FailureOr<CarrierExpansion> expansion =
                appendExpandedValue(builder, forOp.getLoc(), init, initValues, resultTypes, forOp);
            if (failed(expansion))
                return failure();
            expansions.push_back(std::move(*expansion));
        }

        auto newFor = builder.create<scf::ForOp>(
            forOp.getLoc(), forOp.getLowerBound(), forOp.getUpperBound(), forOp.getStep(), initValues);
        newFor->setAttrs(forOp->getAttrs());
        Block* oldBody = forOp.getBody();
        Block* newBody = newFor.getBody();

        forOp.getInductionVar().replaceAllUsesWith(newFor.getInductionVar());
        OpBuilder bodyBuilder(newBody, newBody->begin());
        for (auto [index, expansion] : llvm::enumerate(expansions)) {
            BlockArgument oldArg = forOp.getRegionIterArg(index);
            if (!expansion.isTensor) {
                oldArg.replaceAllUsesWith(newBody->getArgument(1 + expansion.start));
                continue;
            }
            SmallVector<Value, 13> components;
            for (unsigned i = 0; i < expansion.count; ++i)
                components.push_back(newBody->getArgument(1 + expansion.start + i));
            FailureOr<Value> desc = buildDescFromComponents(bodyBuilder, forOp.getLoc(), expansion, components, forOp);
            if (failed(desc))
                return failure();
            oldArg.replaceAllUsesWith(*desc);
        }

        while (!oldBody->empty())
            oldBody->front().moveBefore(newBody, newBody->end());

        if (failed(lowerRegion(newFor.getRegion())))
            return failure();
        auto movedYield = dyn_cast<scf::YieldOp>(newBody->getTerminator());
        if (!movedYield || movedYield.getNumOperands() != expansions.size()) {
            forOp.emitError("tensor-valued scf.for requires one yield per init argument");
            return failure();
        }

        SmallVector<Value, 32> yieldValues;
        OpBuilder yieldBuilder(movedYield);
        for (auto [index, expansion] : llvm::enumerate(expansions)) {
            if (failed(appendUsingExpansion(
                    yieldBuilder, movedYield.getLoc(), movedYield.getOperand(index), expansion, yieldValues, forOp)))
                return failure();
        }
        yieldBuilder.create<scf::YieldOp>(movedYield.getLoc(), yieldValues);
        movedYield.erase();

        OpBuilder resultBuilder(newFor);
        resultBuilder.setInsertionPointAfter(newFor);
        ValueRange flatResults = newFor.getResults();
        SmallVector<Value, 4> replacements;
        for (const CarrierExpansion& expansion : expansions) {
            if (!expansion.isTensor) {
                replacements.push_back(flatResults[expansion.start]);
                continue;
            }
            FailureOr<Value> desc = buildDescFromComponents(
                resultBuilder, forOp.getLoc(), expansion, flatResults.slice(expansion.start, expansion.count), forOp);
            if (failed(desc))
                return failure();
            replacements.push_back(*desc);
        }

        for (auto [oldResult, replacement] : llvm::zip(forOp.getResults(), replacements))
            oldResult.replaceAllUsesWith(replacement);
        forOp.erase();
        return success();
    }

    LogicalResult lowerWhile(scf::WhileOp whileOp)
    {
        if (failed(lowerDerivableTensorProducers()))
            return failure();
        if (whileOp.getInits().size() != whileOp.getNumResults()) {
            whileOp.emitError("tensor-valued scf.while has inconsistent init/result counts");
            return failure();
        }
        for (auto [init, result] : llvm::zip(whileOp.getInits(), whileOp.getResults())) {
            if (init.getType() != result.getType()) {
                whileOp.emitError("tensor-valued scf.while requires stable init/result types");
                return failure();
            }
        }

        SmallVector<Value, 32> initValues;
        SmallVector<Type, 32> resultTypes;
        SmallVector<CarrierExpansion, 4> expansions;
        OpBuilder builder(whileOp);
        for (Value init : whileOp.getInits()) {
            FailureOr<CarrierExpansion> expansion =
                appendExpandedValue(builder, whileOp.getLoc(), init, initValues, resultTypes, whileOp);
            if (failed(expansion))
                return failure();
            expansions.push_back(std::move(*expansion));
        }

        auto emptyBuilder = [](OpBuilder&, Location, ValueRange) {};
        auto newWhile =
            builder.create<scf::WhileOp>(whileOp.getLoc(), resultTypes, initValues, emptyBuilder, emptyBuilder);
        newWhile->setAttrs(whileOp->getAttrs());
        Block* oldBefore = whileOp.getBeforeBody();
        Block* oldAfter = whileOp.getAfterBody();
        Block* newBefore = newWhile.getBeforeBody();
        Block* newAfter = newWhile.getAfterBody();

        OpBuilder beforeBuilder(newBefore, newBefore->begin());
        OpBuilder afterBuilder(newAfter, newAfter->begin());
        for (auto [index, expansion] : llvm::enumerate(expansions)) {
            BlockArgument oldBeforeArg = oldBefore->getArgument(index);
            BlockArgument oldAfterArg = oldAfter->getArgument(index);
            if (!expansion.isTensor) {
                oldBeforeArg.replaceAllUsesWith(newBefore->getArgument(expansion.start));
                oldAfterArg.replaceAllUsesWith(newAfter->getArgument(expansion.start));
                continue;
            }

            SmallVector<Value, 13> beforeComponents;
            SmallVector<Value, 13> afterComponents;
            for (unsigned i = 0; i < expansion.count; ++i) {
                beforeComponents.push_back(newBefore->getArgument(expansion.start + i));
                afterComponents.push_back(newAfter->getArgument(expansion.start + i));
            }
            FailureOr<Value> beforeDesc =
                buildDescFromComponents(beforeBuilder, whileOp.getLoc(), expansion, beforeComponents, whileOp);
            FailureOr<Value> afterDesc =
                buildDescFromComponents(afterBuilder, whileOp.getLoc(), expansion, afterComponents, whileOp);
            if (failed(beforeDesc) || failed(afterDesc))
                return failure();
            oldBeforeArg.replaceAllUsesWith(*beforeDesc);
            oldAfterArg.replaceAllUsesWith(*afterDesc);
        }

        while (!oldBefore->empty())
            oldBefore->front().moveBefore(newBefore, newBefore->end());
        while (!oldAfter->empty())
            oldAfter->front().moveBefore(newAfter, newAfter->end());

        if (failed(lowerRegion(newWhile.getBefore())) || failed(lowerRegion(newWhile.getAfter())))
            return failure();

        auto conditionOp = dyn_cast<scf::ConditionOp>(newBefore->getTerminator());
        auto yieldOp = dyn_cast<scf::YieldOp>(newAfter->getTerminator());
        if (!conditionOp || !yieldOp || conditionOp.getArgs().size() != expansions.size() ||
            yieldOp.getNumOperands() != expansions.size()) {
            whileOp.emitError("tensor-valued scf.while requires matching condition/yield carriers");
            return failure();
        }

        SmallVector<Value, 32> conditionValues;
        OpBuilder conditionBuilder(conditionOp);
        for (auto [index, expansion] : llvm::enumerate(expansions)) {
            if (failed(appendUsingExpansion(
                    conditionBuilder, conditionOp.getLoc(), conditionOp.getArgs()[index], expansion, conditionValues,
                    whileOp)))
                return failure();
        }
        conditionBuilder.create<scf::ConditionOp>(conditionOp.getLoc(), conditionOp.getCondition(), conditionValues);
        conditionOp.erase();

        SmallVector<Value, 32> yieldValues;
        OpBuilder yieldBuilder(yieldOp);
        for (auto [index, expansion] : llvm::enumerate(expansions)) {
            if (failed(appendUsingExpansion(
                    yieldBuilder, yieldOp.getLoc(), yieldOp.getOperand(index), expansion, yieldValues, whileOp)))
                return failure();
        }
        yieldBuilder.create<scf::YieldOp>(yieldOp.getLoc(), yieldValues);
        yieldOp.erase();

        OpBuilder resultBuilder(newWhile);
        resultBuilder.setInsertionPointAfter(newWhile);
        ValueRange flatResults = newWhile.getResults();
        SmallVector<Value, 4> replacements;
        for (const CarrierExpansion& expansion : expansions) {
            if (!expansion.isTensor) {
                replacements.push_back(flatResults[expansion.start]);
                continue;
            }
            FailureOr<Value> desc = buildDescFromComponents(
                resultBuilder, whileOp.getLoc(), expansion, flatResults.slice(expansion.start, expansion.count),
                whileOp);
            if (failed(desc))
                return failure();
            replacements.push_back(*desc);
        }

        for (auto [oldResult, replacement] : llvm::zip(whileOp.getResults(), replacements))
            oldResult.replaceAllUsesWith(replacement);
        whileOp.erase();
        return success();
    }

    func::FuncOp funcOp;
    TensorDescriptorDerivation& derivation;
};

class TlaLowerTensorDescPass : public PassWrapper<TlaLowerTensorDescPass, OperationPass<ModuleOp>> {
public:
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TlaLowerTensorDescPass)

    StringRef getArgument() const override
    {
        return "tla-lower-tensor-desc";
    }
    StringRef getName() const override
    {
        return "TlaLowerTensorDescPass";
    }
    StringRef getDescription() const override
    {
        return "Materialize tensor descriptors and structural SCF carriers.";
    }
    void getDependentDialects(DialectRegistry& registry) const override
    {
        registry.insert<arith::ArithDialect, mlir::memref::MemRefDialect, scf::SCFDialect, ::tla::TlaDialect>();
    }

    void runOnOperation() override
    {
        ModuleOp module = getOperation();
        SmallVector<func::FuncOp, 4> funcOps(module.getOps<func::FuncOp>());
        for (func::FuncOp funcOp : funcOps) {
            if (funcOp.isDeclaration())
                continue;
            ::tla::TensorDescriptorDerivation derivation;
            TensorDescControlFlowLowering materializer(funcOp, derivation);
            if (failed(materializer.run())) {
                signalPassFailure();
                return;
            }
        }
    }
};

} // namespace

std::unique_ptr<Pass> createTlaLowerTensorDescPass()
{
    return std::make_unique<TlaLowerTensorDescPass>();
}

void registerTlaLowerTensorDescPass()
{
    PassRegistration<TlaLowerTensorDescPass>();
}

} // namespace tla
