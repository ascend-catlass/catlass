// -----------------------------------------------------------------------------------------------------------
// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// -----------------------------------------------------------------------------------------------------------

#include "PassesCommon.h"
#include "PassesInternal.h"

namespace tla {
namespace {

enum class SplitSide
{
    AIC,
    AIV
};

static bool isMixedSplitCandidate(func::FuncOp funcOp)
{
    if (funcOp.isDeclaration() || isPrivateSymbol(funcOp))
        return false;
    auto coreType = funcOp->getAttrOfType<hivm::TFuncCoreTypeAttr>(hivm::TFuncCoreTypeAttr::name);
    return coreType && coreType.getFuncCoreType() == hivm::TFuncCoreType::MIX;
}

static bool isCoreRegionWrapper(Operation* op)
{
    return isa<::tla::CubeOp, ::tla::VectorOp>(op);
}

static bool isNestedWithin(Operation* op, Operation* ancestor)
{
    for (Operation* current = op; current; current = current->getParentOp()) {
        if (current == ancestor)
            return true;
    }
    return false;
}

static LogicalResult validateValueUsesWithinScope(Value value, Operation* scope)
{
    for (Operation* user : value.getUsers()) {
        if (!isNestedWithin(user, scope)) {
            return scope->emitOpError() << "defines an SSA value that escapes its scope to operation '"
                                        << user->getName().getStringRef() << "'";
        }
    }
    return success();
}

static LogicalResult validateScopeValuesDoNotEscape(Operation* scope)
{
    LogicalResult result = success();
    scope->walk([&](Operation* nested) {
        for (Value value : nested->getResults()) {
            if (failed(validateValueUsesWithinScope(value, scope))) {
                result = failure();
                return WalkResult::interrupt();
            }
        }
        for (Region& region : nested->getRegions()) {
            for (Block& block : region) {
                for (BlockArgument argument : block.getArguments()) {
                    if (failed(validateValueUsesWithinScope(argument, scope))) {
                        result = failure();
                        return WalkResult::interrupt();
                    }
                }
            }
        }
        return WalkResult::advance();
    });
    return result;
}

static LogicalResult validateMixedFunction(func::FuncOp funcOp)
{
    if (funcOp.getBody().empty())
        return funcOp.emitOpError() << "mixed function must have a body";

    bool hasCube = false;
    bool hasVector = false;
    LogicalResult result = success();
    funcOp.walk([&](Operation* op) {
        if (!isCoreRegionWrapper(op))
            return WalkResult::advance();

        hasCube |= isa<::tla::CubeOp>(op);
        hasVector |= isa<::tla::VectorOp>(op);
        for (Operation* parent = op->getParentOp(); parent && parent != funcOp.getOperation();
             parent = parent->getParentOp()) {
            if (isCoreRegionWrapper(parent)) {
                result = op->emitOpError() << "tla.cube and tla.vector scopes must not be nested in a mixed kernel";
                return WalkResult::interrupt();
            }
        }

        if (failed(validateScopeValuesDoNotEscape(op))) {
            result = failure();
            return WalkResult::interrupt();
        }
        return WalkResult::advance();
    });
    if (failed(result))
        return failure();

    if (!hasCube || !hasVector) {
        return funcOp.emitOpError() << "MIX function requires at least one tla.cube and one tla.vector scope";
    }
    return success();
}

static LogicalResult validateSplitSymbolNames(ModuleOp module, func::FuncOp funcOp)
{
    SymbolTable symbolTable(module);
    for (StringRef suffix : {StringRef("_mix_aic"), StringRef("_mix_aiv")}) {
        std::string splitName = (funcOp.getSymName() + suffix).str();
        if (symbolTable.lookup(splitName)) {
            return funcOp.emitOpError() << "cannot split mixed function because symbol @" << splitName
                                        << " already exists";
        }
    }
    return success();
}

static bool containsCubeRegion(func::FuncOp funcOp)
{
    return funcOp.walk([&](::tla::CubeOp) { return WalkResult::interrupt(); }).wasInterrupted();
}

static bool containsVectorRegion(func::FuncOp funcOp)
{
    return funcOp.walk([&](::tla::VectorOp) { return WalkResult::interrupt(); }).wasInterrupted();
}

static void eraseOppositeScopes(func::FuncOp funcOp, SplitSide side)
{
    SmallVector<Operation*, 8> eraseList;
    if (side == SplitSide::AIC) {
        funcOp.walk([&](::tla::VectorOp vectorOp) { eraseList.push_back(vectorOp); });
    } else {
        funcOp.walk([&](::tla::CubeOp cubeOp) { eraseList.push_back(cubeOp); });
    }
    for (Operation* op : llvm::reverse(eraseList))
        op->erase();
}

static void eraseUnusedSplitFlags(func::FuncOp funcOp)
{
    SmallVector<Operation*, 8> eraseList;
    funcOp.walk([&](::tla::FlagOp flagOp) {
        if (flagOp->use_empty())
            eraseList.push_back(flagOp);
    });
    for (Operation* op : eraseList)
        op->erase();
}

static func::FuncOp createSplitFunction(func::FuncOp source, StringRef newName, SplitSide side)
{
    auto newFunc = cast<func::FuncOp>(source->clone());
    newFunc.setSymName(newName);

    MLIRContext* ctx = source.getContext();
    hivm::TFuncCoreType coreType = side == SplitSide::AIC ? hivm::TFuncCoreType::AIC : hivm::TFuncCoreType::AIV;
    setRequiredHaccEntryAttrs(newFunc, ctx);
    setC310RegbaseTargetAttr(newFunc, ctx);
    newFunc->setAttr(hivm::TFuncCoreTypeAttr::name, hivm::TFuncCoreTypeAttr::get(ctx, coreType));
    newFunc->setAttr(kMixModeAttrName, StringAttr::get(ctx, "mix"));
    newFunc->setAttr(kParallelModeAttrName, StringAttr::get(ctx, "simd"));
    newFunc->setAttr(hivm::TPartOfMixAttr::name, UnitAttr::get(ctx));
    newFunc->setAttr(hivm::VFModeAttr::name, hivm::VFModeAttr::get(ctx, hivm::VFMode::SIMD));
    // Clone already copied ``tla.core_ratio`` / ``hivm.core_ratio``; re-set
    // explicitly so split halves stay aligned if stamp order changes.
    if (auto mixRatio = source->getAttr(kCoreRatioAttrName))
        newFunc->setAttr(kCoreRatioAttrName, mixRatio);
    if (auto coreRatio = source->getAttr(hivm::TCoreRatioAttr::name))
        newFunc->setAttr(hivm::TCoreRatioAttr::name, coreRatio);

    eraseOppositeScopes(newFunc, side);
    eraseUnusedSplitFlags(newFunc);
    return newFunc;
}

static LogicalResult validateSplitPostconditions(func::FuncOp source, func::FuncOp aicFunc, func::FuncOp aivFunc)
{
    if (containsVectorRegion(aicFunc) || !containsCubeRegion(aicFunc)) {
        return source.emitOpError() << "failed to prepare AIC split: expected cube scopes and no vector scopes";
    }
    if (containsCubeRegion(aivFunc) || !containsVectorRegion(aivFunc)) {
        return source.emitOpError() << "failed to prepare AIV split: expected vector scopes and no cube scopes";
    }
    return success();
}

class TlaSplitMixedFuncPass : public PassWrapper<TlaSplitMixedFuncPass, OperationPass<ModuleOp>> {
public:
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TlaSplitMixedFuncPass)

    StringRef getArgument() const override
    {
        return "tla-split-mixed-func";
    }
    StringRef getName() const override
    {
        return "TlaSplitMixedFuncPass";
    }
    StringRef getDescription() const override
    {
        return "Materialize frontend-declared mixed Tla functions into AIC and AIV functions.";
    }

    void getDependentDialects(DialectRegistry& registry) const override
    {
        registry.insert<
            func::FuncDialect, ::tla::TlaDialect, hacc::HACCDialect, hivm::HIVMDialect,
            hivm_regbaseintrins::HIVMRegbaseIntrinsDialect>();
    }

    void runOnOperation() override
    {
        ModuleOp module = getOperation();
        SmallVector<func::FuncOp, 4> candidates;
        for (func::FuncOp funcOp : module.getOps<func::FuncOp>()) {
            if (isMixedSplitCandidate(funcOp))
                candidates.push_back(funcOp);
        }

        // Validate every MIX function before modifying the module, so an invalid
        // function cannot leave earlier candidates partially split.
        for (func::FuncOp funcOp : candidates) {
            if (failed(validateMixedFunction(funcOp)) || failed(validateSplitSymbolNames(module, funcOp))) {
                signalPassFailure();
                return;
            }
        }

        for (func::FuncOp funcOp : candidates) {
            std::string aicName = (funcOp.getSymName() + "_mix_aic").str();
            std::string aivName = (funcOp.getSymName() + "_mix_aiv").str();
            func::FuncOp aicFunc = createSplitFunction(funcOp, aicName, SplitSide::AIC);
            func::FuncOp aivFunc = createSplitFunction(funcOp, aivName, SplitSide::AIV);

            // Both detached clones must be valid before either is inserted and the
            // source symbol is replaced.
            if (failed(validateSplitPostconditions(funcOp, aicFunc, aivFunc))) {
                aicFunc->destroy();
                aivFunc->destroy();
                signalPassFailure();
                return;
            }

            OpBuilder builder(funcOp);
            builder.insert(aicFunc);
            builder.insert(aivFunc);
            funcOp.erase();
        }
    }
};

} // namespace

std::unique_ptr<Pass> createTlaSplitMixedFuncPass()
{
    return std::make_unique<TlaSplitMixedFuncPass>();
}

void registerTlaSplitMixedFuncPass()
{
    PassRegistration<TlaSplitMixedFuncPass>();
}

} // namespace tla
