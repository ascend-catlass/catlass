#include "PassesCommon.h"
#include "PassesInternal.h"
#include "TlaScratchAllocation.h"

#include "mlir/IR/Matchers.h"
#include "llvm/ADT/STLExtras.h"

#include <array>
#include <memory>

namespace tla {
namespace {

inline constexpr StringLiteral kAutoSyncAttrName = "tla.auto_sync";
inline constexpr unsigned kMaxAutoMutexIds = 32;

enum class MutexIdSpace : unsigned
{
    Cube = 0,
    Vector = 1,
};

static constexpr unsigned mutexIdSpaceIndex(MutexIdSpace space)
{
    return static_cast<unsigned>(space);
}

static StringRef stringifyMutexIdSpace(MutexIdSpace space)
{
    switch (space) {
        case MutexIdSpace::Cube:
            return "cube";
        case MutexIdSpace::Vector:
            return "vector";
    }
    llvm_unreachable("unknown mutex ID space");
}

struct StorageExpr;
using StorageExprPtr = std::shared_ptr<StorageExpr>;

struct StorageExpr {
    enum class Kind
    {
        Global,
        Root,
        Select,
        Unknown
    } kind;
    Value root;
    Value condition;
    StorageExprPtr thenValue;
    StorageExprPtr elseValue;
    Value selectorSource;
    std::array<Value, 2> materializedMutexes;

    static StorageExprPtr global()
    {
        return std::make_shared<StorageExpr>(StorageExpr{Kind::Global, {}, {}, {}, {}, {}, {}});
    }
    static StorageExprPtr rootValue(Value root)
    {
        return std::make_shared<StorageExpr>(StorageExpr{Kind::Root, root, {}, {}, {}, {}, {}});
    }
    static StorageExprPtr unknown()
    {
        return std::make_shared<StorageExpr>(StorageExpr{Kind::Unknown, {}, {}, {}, {}, {}, {}});
    }
};

static bool sameStorageExpr(const StorageExprPtr& lhs, const StorageExprPtr& rhs)
{
    if (!lhs || !rhs || lhs->kind != rhs->kind)
        return false;
    switch (lhs->kind) {
        case StorageExpr::Kind::Global:
        case StorageExpr::Kind::Unknown:
            return true;
        case StorageExpr::Kind::Root:
            return lhs->root == rhs->root;
        case StorageExpr::Kind::Select:
            return lhs->condition == rhs->condition && sameStorageExpr(lhs->thenValue, rhs->thenValue) &&
                   sameStorageExpr(lhs->elseValue, rhs->elseValue);
    }
    llvm_unreachable("unknown storage expression kind");
}

static StorageExprPtr selectStorage(
    Value selectorSource, Value condition, StorageExprPtr thenValue, StorageExprPtr elseValue)
{
    if (sameStorageExpr(thenValue, elseValue))
        return thenValue;
    if (!thenValue || !elseValue || thenValue->kind == StorageExpr::Kind::Unknown ||
        elseValue->kind == StorageExpr::Kind::Unknown || thenValue->kind == StorageExpr::Kind::Global ||
        elseValue->kind == StorageExpr::Kind::Global)
        return StorageExpr::unknown();
    return std::make_shared<StorageExpr>(StorageExpr{
        StorageExpr::Kind::Select, {}, condition, std::move(thenValue), std::move(elseValue), selectorSource, {}});
}

class StorageResolver {
public:
    explicit StorageResolver(const TlaScratchAllocationPlan& allocationPlan) : allocationPlan(allocationPlan)
    {}

    StorageExprPtr resolve(Value value)
    {
        DenseMap<Value, StorageExprPtr> assumptions;
        DenseSet<Value> visiting;
        return resolveImpl(value, assumptions, visiting);
    }

private:
    StorageExprPtr resolveScfResult(
        Value value, DenseMap<Value, StorageExprPtr>& assumptions, DenseSet<Value>& visiting)
    {
        auto result = dyn_cast<OpResult>(value);
        if (!result)
            return StorageExpr::unknown();

        if (auto ifOp = value.getDefiningOp<scf::IfOp>()) {
            unsigned index = result.getResultNumber();
            scf::YieldOp thenYield = ifOp.thenYield();
            scf::YieldOp elseYield = ifOp.elseYield();
            if (!thenYield || !elseYield || index >= thenYield.getNumOperands() || index >= elseYield.getNumOperands())
                return StorageExpr::unknown();
            DenseSet<Value> thenVisiting = visiting;
            DenseSet<Value> elseVisiting = visiting;
            StorageExprPtr thenValue = resolveImpl(thenYield.getOperand(index), assumptions, thenVisiting);
            StorageExprPtr elseValue = resolveImpl(elseYield.getOperand(index), assumptions, elseVisiting);
            return selectStorage(value, ifOp.getCondition(), std::move(thenValue), std::move(elseValue));
        }

        if (auto forOp = value.getDefiningOp<scf::ForOp>()) {
            unsigned index = result.getResultNumber();
            if (index >= forOp.getInitArgs().size())
                return StorageExpr::unknown();
            DenseSet<Value> initVisiting = visiting;
            StorageExprPtr init = resolveImpl(forOp.getInitArgs()[index], assumptions, initVisiting);
            if (init->kind == StorageExpr::Kind::Unknown)
                return init;
            scf::YieldOp yield = dyn_cast<scf::YieldOp>(forOp.getBody()->getTerminator());
            if (!yield || index >= yield.getNumOperands())
                return StorageExpr::unknown();
            Value iterArg = forOp.getRegionIterArg(index);
            if (yield.getOperand(index) == iterArg)
                return init;
            DenseMap<Value, StorageExprPtr> loopAssumptions = assumptions;
            loopAssumptions[iterArg] = init;
            DenseSet<Value> yieldVisiting = visiting;
            StorageExprPtr backedge = resolveImpl(yield.getOperand(index), loopAssumptions, yieldVisiting);
            return sameStorageExpr(init, backedge) ? init : StorageExpr::unknown();
        }
        return StorageExpr::unknown();
    }

    StorageExprPtr resolveForBlockArgument(
        BlockArgument blockArg, DenseMap<Value, StorageExprPtr>& assumptions, DenseSet<Value>& visiting)
    {
        auto assumed = assumptions.find(blockArg);
        if (assumed != assumptions.end())
            return assumed->second;
        Operation* parent = blockArg.getOwner()->getParentOp();
        auto forOp = dyn_cast_or_null<scf::ForOp>(parent);
        unsigned argNumber = blockArg.getArgNumber();
        if (!forOp || blockArg.getOwner() != forOp.getBody() || argNumber == 0 ||
            argNumber - 1 >= forOp.getInitArgs().size())
            return StorageExpr::unknown();
        unsigned index = argNumber - 1;
        DenseSet<Value> initVisiting = visiting;
        StorageExprPtr init = resolveImpl(forOp.getInitArgs()[index], assumptions, initVisiting);
        if (init->kind == StorageExpr::Kind::Unknown)
            return init;
        scf::YieldOp yield = dyn_cast<scf::YieldOp>(forOp.getBody()->getTerminator());
        if (!yield || index >= yield.getNumOperands())
            return StorageExpr::unknown();
        if (yield.getOperand(index) == blockArg)
            return init;
        DenseMap<Value, StorageExprPtr> loopAssumptions = assumptions;
        loopAssumptions[blockArg] = init;
        DenseSet<Value> yieldVisiting = visiting;
        StorageExprPtr backedge = resolveImpl(yield.getOperand(index), loopAssumptions, yieldVisiting);
        return sameStorageExpr(init, backedge) ? init : StorageExpr::unknown();
    }

    StorageExprPtr resolveImpl(Value value, DenseMap<Value, StorageExprPtr>& assumptions, DenseSet<Value>& visiting)
    {
        if (!value)
            return StorageExpr::unknown();
        if (auto assumed = assumptions.find(value); assumed != assumptions.end())
            return assumed->second;
        if (!visiting.insert(value).second)
            return StorageExpr::unknown();

        if (isa<MemRefType>(value.getType()))
            return StorageExpr::global();
        if (auto blockArg = dyn_cast<BlockArgument>(value))
            return resolveForBlockArgument(blockArg, assumptions, visiting);
        if (isa<OpResult>(value) && isa_and_nonnull<scf::IfOp, scf::ForOp>(value.getDefiningOp()))
            return resolveScfResult(value, assumptions, visiting);

        if (auto alloc = value.getDefiningOp<::tla::AllocPtrOp>()) {
            if (!allocationPlan.lookup(alloc.getResult()))
                return StorageExpr::unknown();
            return StorageExpr::rootValue(alloc.getResult());
        }
        if (auto recast = value.getDefiningOp<::tla::RecastPtrOp>())
            return resolveImpl(recast.getSrc(), assumptions, visiting);
        if (auto add = value.getDefiningOp<::tla::PtrAddOp>())
            return resolveImpl(add.getPtr(), assumptions, visiting);
        if (auto tensorPtr = value.getDefiningOp<::tla::TensorPtrOp>())
            return resolveImpl(tensorPtr.getSrc(), assumptions, visiting);
        // UPDATE next_ptr aliases the same UB root as the window it advanced
        // from — required so auto-sync sees outlined chained UPDATE.
        if (auto loadUpdate = value.getDefiningOp<::tla::LoadUpdateOp>())
            return resolveImpl(loadUpdate.getSource(), assumptions, visiting);
        if (auto storeUpdate = value.getDefiningOp<::tla::StoreUpdateOp>())
            return resolveImpl(storeUpdate.getDest(), assumptions, visiting);
        if (auto loadUnalignUpdate = value.getDefiningOp<::tla::LoadUnalignUpdateOp>())
            return resolveImpl(loadUnalignUpdate.getSource(), assumptions, visiting);
        if (auto storeUnalignUpdate = value.getDefiningOp<::tla::StoreUnalignUpdateOp>())
            return resolveImpl(storeUnalignUpdate.getDest(), assumptions, visiting);
        if (auto ptrToInt = value.getDefiningOp<::tla::PtrToIntOp>())
            return resolveImpl(ptrToInt.getPtr(), assumptions, visiting);
        if (auto intToPtr = value.getDefiningOp<::tla::IntToPtrOp>()) {
            auto ptrType = dyn_cast<::tla::PtrType>(intToPtr.getResult().getType());
            if (ptrType &&
                (ptrType.getAddrspace() == AddressSpace::gm || ptrType.getAddrspace() == AddressSpace::generic))
                return StorageExpr::global();
            // Prefer the defining ptr when the address is ``ptrtoint`` of one;
            // otherwise a bare on-chip i64 has no allocation root.
            if (auto fromPtr = intToPtr.getAddr().getDefiningOp<::tla::PtrToIntOp>())
                return resolveImpl(fromPtr.getPtr(), assumptions, visiting);
            return StorageExpr::unknown();
        }
        if (auto desc = value.getDefiningOp<::tla::TensorDescOp>())
            return resolveImpl(desc.getBase(), assumptions, visiting);
        if (auto tile = value.getDefiningOp<::tla::TileViewOp>())
            return resolveImpl(tile.getSource(), assumptions, visiting);
        if (auto make = value.getDefiningOp<::tla::MakeTensorOp>())
            return resolveImpl(make.getPtr(), assumptions, visiting);
        if (auto makeLike = value.getDefiningOp<::tla::MakeTensorLikeOp>())
            return resolveImpl(makeLike.getPtr(), assumptions, visiting);
        if (auto cast = value.getDefiningOp<UnrealizedConversionCastOp>()) {
            if (cast.getNumOperands() == 1)
                return resolveImpl(cast.getOperand(0), assumptions, visiting);
        }

        if (auto ptrType = dyn_cast<::tla::PtrType>(value.getType())) {
            if (ptrType.getAddrspace() == AddressSpace::gm || ptrType.getAddrspace() == AddressSpace::generic)
                return StorageExpr::global();
        }
        return StorageExpr::unknown();
    }

    const TlaScratchAllocationPlan& allocationPlan;
};

struct InstructionPlan {
    Operation* op;
    ::Pipe pipe;
    MutexIdSpace idSpace;
    SmallVector<StorageExprPtr, 8> resources;
};

static FailureOr<MutexIdSpace> resolveMutexIdSpace(func::FuncOp func, Operation* instruction)
{
    for (Operation* parent = instruction->getParentOp(); parent && parent != func.getOperation();
         parent = parent->getParentOp()) {
        if (isa<::tla::CubeOp>(parent))
            return MutexIdSpace::Cube;
        if (isa<::tla::VectorOp>(parent))
            return MutexIdSpace::Vector;
    }

    std::optional<hivm::TFuncCoreType> coreType = getFunctionCoreType(func.getOperation());
    if (coreType == hivm::TFuncCoreType::AIC)
        return MutexIdSpace::Cube;
    if (coreType == hivm::TFuncCoreType::AIV)
        return MutexIdSpace::Vector;

    instruction->emitError(
        "automatic mutex cannot determine the core-local mutex ID space; "
        "instructions in a MIX kernel must be inside tla.cube or tla.vector");
    return failure();
}

static unsigned addressSpaceRank(::AddressSpace space)
{
    switch (space) {
        case AddressSpace::l1:
            return 0;
        case AddressSpace::l0a:
            return 1;
        case AddressSpace::l0b:
            return 2;
        case AddressSpace::l0c:
            return 3;
        case AddressSpace::ub:
            return 4;
        case AddressSpace::biasbuf:
            return 6;
        case AddressSpace::generic:
        case AddressSpace::gm:
            return 7;
    }
    llvm_unreachable("unknown TLA address space");
}

// Takes the source tile rather than the op: tla.copy and tla.copy_mx pick their
// pipe the same way, from where the data is coming from.
static FailureOr<::Pipe> inferCopyPipe(Value src)
{
    auto sourceType = dyn_cast<::tla::TlaTensorType>(src.getType());
    if (!sourceType)
        return failure();
    switch (sourceType.getPtr().getAddrspace()) {
        case AddressSpace::gm:
            return Pipe::mte2;
        case AddressSpace::l1:
            return Pipe::mte1;
        case AddressSpace::ub:
            return Pipe::mte3;
        case AddressSpace::l0c:
            return Pipe::fix;
        case AddressSpace::generic:
        case AddressSpace::l0a:
        case AddressSpace::l0b:
        case AddressSpace::biasbuf:
            return failure();
    }
    return failure();
}

static void collectCandidateRoots(const StorageExprPtr& expr, DenseSet<Value>& roots)
{
    if (!expr)
        return;
    if (expr->kind == StorageExpr::Kind::Root) {
        roots.insert(expr->root);
        return;
    }
    if (expr->kind == StorageExpr::Kind::Select) {
        collectCandidateRoots(expr->thenValue, roots);
        collectCandidateRoots(expr->elseValue, roots);
    }
}

static bool haveOverlappingCandidates(const StorageExprPtr& lhs, const StorageExprPtr& rhs)
{
    DenseSet<Value> lhsRoots;
    DenseSet<Value> rhsRoots;
    collectCandidateRoots(lhs, lhsRoots);
    collectCandidateRoots(rhs, rhsRoots);
    return llvm::any_of(lhsRoots, [&](Value root) { return rhsRoots.contains(root); });
}

static LogicalResult validateNoManualLocalSync(func::FuncOp func)
{
    Operation* invalid = nullptr;
    func.walk([&](Operation* op) {
        if (!invalid && isa<::tla::MutexOp, ::tla::MutexLockOp, ::tla::MutexUnlockOp, ::tla::FlagOp, ::tla::SetFlagOp,
                            ::tla::WaitFlagOp>(op))
            invalid = op;
    });
    if (!invalid)
        return success();
    return invalid->emitError(
        "auto_sync='v0' cannot be combined with local mutex, mutex_guard, "
        "or local flag synchronization; cross_core_* remains explicit");
}

// In TLA DSL extern op support v1, the user is responsible for ensuring that
// external calls are properly synchronized. we do not attempt to automatically
// synchronize them.
static LogicalResult validateNoExternCalls(func::FuncOp func)
{
    ::tla::CallExternOp invalid;
    func.walk([&](::tla::CallExternOp op) {
        if (!invalid)
            invalid = op;
    });
    if (!invalid)
        return success();
    return invalid.emitError(
        "auto_sync='v0' cannot be combined with tla.call_extern; "
        "external calls require explicit synchronization in v1");
}

static LogicalResult validateCopyUnitFlags(func::FuncOp func)
{
    LogicalResult result = success();
    func.walk([&](::tla::CopyL0C2DstParamsOp op) {
        if (op.getUnitFlag() != 0 && op.getUnitFlag() != 3) {
            op.emitError("auto_sync='v0' supports copy unit_flag values 0 or 3");
            result = failure();
        }
    });
    return result;
}

enum UnitFlagPossibility : unsigned
{
    UnitFlagZero = 1U << 0,
    UnitFlagEnabled = 1U << 1,
    UnitFlagInvalid = 1U << 2,
    UnitFlagUnknown = 1U << 3,
};

static unsigned analyzeUnitFlagPossibilities(Value value, DenseSet<Value>& visiting)
{
    APInt constant;
    if (matchPattern(value, m_ConstantInt(&constant))) {
        if (constant.isZero())
            return UnitFlagZero;
        if (constant == 2 || constant == 3)
            return UnitFlagEnabled;
        return UnitFlagInvalid;
    }

    if (!visiting.insert(value).second)
        return UnitFlagUnknown;

    unsigned possibilities = UnitFlagUnknown;
    Operation* definingOp = value.getDefiningOp();
    if (definingOp && isa<arith::ExtSIOp, arith::ExtUIOp, arith::IndexCastOp>(definingOp)) {
        possibilities = analyzeUnitFlagPossibilities(definingOp->getOperand(0), visiting);
    } else if (auto select = dyn_cast_or_null<arith::SelectOp>(definingOp)) {
        possibilities = analyzeUnitFlagPossibilities(select.getTrueValue(), visiting) |
                        analyzeUnitFlagPossibilities(select.getFalseValue(), visiting);
    } else if (auto ifOp = dyn_cast_or_null<scf::IfOp>(definingOp)) {
        auto result = dyn_cast<OpResult>(value);
        scf::YieldOp thenYield = ifOp.thenYield();
        scf::YieldOp elseYield = ifOp.elseYield();
        if (result && thenYield && elseYield && result.getResultNumber() < thenYield.getNumOperands() &&
            result.getResultNumber() < elseYield.getNumOperands()) {
            unsigned index = result.getResultNumber();
            possibilities = analyzeUnitFlagPossibilities(thenYield.getOperand(index), visiting) |
                            analyzeUnitFlagPossibilities(elseYield.getOperand(index), visiting);
        }
    }

    visiting.erase(value);
    return possibilities;
}

// Takes the op generically so both tla.mmad and tla.mmad_mx are covered: they
// carry the same unit_flag operand and must both take part in auto-mutex.
static FailureOr<bool> isMmadUnitFlagEnabled(Operation* op, Value unitFlag)
{
    DenseSet<Value> visiting;
    unsigned possibilities = analyzeUnitFlagPossibilities(unitFlag, visiting);
    if (possibilities == UnitFlagZero)
        return false;
    if (possibilities == UnitFlagEnabled)
        return true;
    if (possibilities & UnitFlagInvalid) {
        op->emitError("auto_sync='v0' supports tla.mmad unit_flag values 0, 2, or 3");
        return failure();
    }
    if ((possibilities & UnitFlagZero) && (possibilities & UnitFlagEnabled)) {
        op->emitError(
            "auto_sync='v0' requires tla.mmad unit_flag to be always zero or always enabled; "
            "a runtime choice between zero and 2/3 cannot safely determine L0C locking");
        return failure();
    }
    op->emitError(
        "auto_sync='v0' requires tla.mmad unit_flag to be provably always zero or always enabled with value 2/3");
    return failure();
}

static LogicalResult collectInstructionPlans(
    func::FuncOp func, StorageResolver& resolver, SmallVectorImpl<InstructionPlan>& plans)
{
    LogicalResult result = success();
    func.walk([&](Operation* operation) {
        if (failed(result))
            return;
        if (auto copy = dyn_cast<::tla::CopyOp>(operation)) {
            FailureOr<::Pipe> pipe = inferCopyPipe(copy.getSrc());
            if (failed(pipe)) {
                copy.emitError("cannot infer automatic mutex pipe for tla.copy");
                result = failure();
                return;
            }
            FailureOr<MutexIdSpace> idSpace = resolveMutexIdSpace(func, copy);
            if (failed(idSpace)) {
                result = failure();
                return;
            }
            InstructionPlan plan{copy, *pipe, *idSpace, {}};
            plan.resources.push_back(resolver.resolve(copy.getDst()));
            bool unitFlagEnabled = false;
            if (Value params = copy.getParams()) {
                if (auto copyParams = params.getDefiningOp<::tla::CopyL0C2DstParamsOp>()) {
                    unitFlagEnabled = copyParams.getUnitFlag() == 3;
                    if (Value quant = copyParams.getQuantTensor(); quant && isa<::tla::TlaTensorType>(quant.getType()))
                        plan.resources.push_back(resolver.resolve(quant));
                }
            }
            auto sourceType = dyn_cast<::tla::TlaTensorType>(copy.getSrc().getType());
            bool omitUnitFlagL0C =
                unitFlagEnabled && sourceType && sourceType.getPtr().getAddrspace() == AddressSpace::l0c;
            if (!omitUnitFlagL0C)
                plan.resources.push_back(resolver.resolve(copy.getSrc()));
            plans.push_back(std::move(plan));
            return;
        }
        // tla.copy_mx is an L1 -> L0A/L0B load like the tla.copy above, and needs
        // the same treatment: without it an MX kernel's operand loads sit outside
        // auto-sync entirely. It carries one resource the plain copy has no
        // equivalent of -- the e8m0 scale tile, read from L1 alongside the
        // operand -- so all three tiles go into the plan.
        if (auto copyMx = dyn_cast<::tla::CopyMxOp>(operation)) {
            FailureOr<::Pipe> pipe = inferCopyPipe(copyMx.getSrc());
            if (failed(pipe)) {
                copyMx.emitError("cannot infer automatic mutex pipe for tla.copy_mx");
                result = failure();
                return;
            }
            FailureOr<MutexIdSpace> idSpace = resolveMutexIdSpace(func, copyMx);
            if (failed(idSpace)) {
                result = failure();
                return;
            }
            InstructionPlan plan{copyMx, *pipe, *idSpace, {}};
            plan.resources.push_back(resolver.resolve(copyMx.getDst()));
            plan.resources.push_back(resolver.resolve(copyMx.getSrc()));
            plan.resources.push_back(resolver.resolve(copyMx.getScale()));
            plans.push_back(std::move(plan));
            return;
        }
        // tla.img2col is an L1 -> L0A load on the same MTE1 pipe as the plain
        // tla.copy that does the non-convolution version of the same hop, so it
        // needs the same plan -- otherwise a conv kernel's operand loads sit
        // outside auto-sync entirely.
        if (auto img2col = dyn_cast<::tla::Img2colOp>(operation)) {
            FailureOr<::Pipe> pipe = inferCopyPipe(img2col.getSrc());
            if (failed(pipe)) {
                img2col.emitError("cannot infer automatic mutex pipe for tla.img2col");
                result = failure();
                return;
            }
            FailureOr<MutexIdSpace> idSpace = resolveMutexIdSpace(func, img2col);
            if (failed(idSpace)) {
                result = failure();
                return;
            }
            InstructionPlan plan{img2col, *pipe, *idSpace, {}};
            plan.resources.push_back(resolver.resolve(img2col.getDst()));
            plan.resources.push_back(resolver.resolve(img2col.getSrc()));
            plans.push_back(std::move(plan));
            return;
        }
        // AscendC::Fill writes L1 on MTE2; lock only the destination buffer.
        if (auto fill = dyn_cast<::tla::FillOp>(operation)) {
            FailureOr<MutexIdSpace> idSpace = resolveMutexIdSpace(func, fill);
            if (failed(idSpace)) {
                result = failure();
                return;
            }
            InstructionPlan plan{fill, Pipe::mte2, *idSpace, {}};
            plan.resources.push_back(resolver.resolve(fill.getDst()));
            plans.push_back(std::move(plan));
            return;
        }
        // tla.mmad and tla.mmad_mx are treated identically here: same cube pipe,
        // same acc/lhs/rhs/bias resources, same unit_flag rule. Handled through
        // the common accessors so the MX flavour cannot silently skip auto-mutex.
        if (isa<::tla::MmadOp, ::tla::MmadMxOp>(operation)) {
            Value mmAcc;
            Value mmLhs;
            Value mmRhs;
            Value mmBias;
            Value mmUnitFlag;
            if (auto mm = dyn_cast<::tla::MmadOp>(operation)) {
                mmAcc = mm.getAcc();
                mmLhs = mm.getLhs();
                mmRhs = mm.getRhs();
                mmBias = mm.getBias();
                mmUnitFlag = mm.getUnitFlag();
            } else {
                auto mmx = cast<::tla::MmadMxOp>(operation);
                mmAcc = mmx.getAcc();
                mmLhs = mmx.getLhs();
                mmRhs = mmx.getRhs();
                mmBias = mmx.getBias();
                mmUnitFlag = mmx.getUnitFlag();
            }
            FailureOr<MutexIdSpace> idSpace = resolveMutexIdSpace(func, operation);
            if (failed(idSpace)) {
                result = failure();
                return;
            }
            FailureOr<bool> unitFlagEnabled = isMmadUnitFlagEnabled(operation, mmUnitFlag);
            if (failed(unitFlagEnabled)) {
                result = failure();
                return;
            }
            InstructionPlan plan{operation, Pipe::cube, *idSpace, {}};
            if (!*unitFlagEnabled)
                plan.resources.push_back(resolver.resolve(mmAcc));
            plan.resources.push_back(resolver.resolve(mmLhs));
            plan.resources.push_back(resolver.resolve(mmRhs));
            // The bias table is an L0-class operand exactly like lhs/rhs: the
            // cube reads it for the same instruction, so the MTE1 copy that
            // fills biasbuf and the next reload of it both have to be ordered
            // against that read. unit_flag only covers acc, never the bias.
            if (mmBias)
                plan.resources.push_back(resolver.resolve(mmBias));
            plans.push_back(std::move(plan));
            return;
        }
        if (auto vec = dyn_cast<::tla::VecFuncOp>(operation)) {
            FailureOr<MutexIdSpace> idSpace = resolveMutexIdSpace(func, vec);
            if (failed(idSpace)) {
                result = failure();
                return;
            }
            InstructionPlan plan{vec, Pipe::vector, *idSpace, {}};
            vec.walk([&](Operation* nested) {
                if (auto load = dyn_cast<::tla::LoadOp>(nested))
                    plan.resources.push_back(resolver.resolve(load.getSource()));
                else if (auto loadUpdate = dyn_cast<::tla::LoadUpdateOp>(nested))
                    plan.resources.push_back(resolver.resolve(loadUpdate.getSource()));
                else if (auto loadUnalignUpdate = dyn_cast<::tla::LoadUnalignUpdateOp>(nested))
                    plan.resources.push_back(resolver.resolve(loadUnalignUpdate.getSource()));
                else if (auto store = dyn_cast<::tla::StoreOp>(nested))
                    plan.resources.push_back(resolver.resolve(store.getDest()));
                else if (auto storeUpdate = dyn_cast<::tla::StoreUpdateOp>(nested))
                    plan.resources.push_back(resolver.resolve(storeUpdate.getDest()));
                else if (auto storeUnalignUpdate = dyn_cast<::tla::StoreUnalignUpdateOp>(nested))
                    plan.resources.push_back(resolver.resolve(storeUnalignUpdate.getDest()));
                else if (auto gather = dyn_cast<::tla::GatherOp>(nested)) {
                    // Reg→Reg gather has no UB storage operand.
                    if (isa<::tla::TlaTensorType>(gather.getX().getType()))
                        plan.resources.push_back(resolver.resolve(gather.getX()));
                } else if (auto scalarLoad = dyn_cast<::tla::ScalarLoadOp>(nested))
                    plan.resources.push_back(resolver.resolve(scalarLoad.getSource()));
                else if (auto scalarStore = dyn_cast<::tla::ScalarStoreOp>(nested))
                    plan.resources.push_back(resolver.resolve(scalarStore.getDest()));
            });
            plans.push_back(std::move(plan));
            return;
        }
        // UB scalar accesses are legal directly under tla.vector, but the
        // first auto-sync contract deliberately treats tla.vec.func as the
        // vector instruction boundary.  Diagnose the uncovered form instead
        // of silently compiling a local access without a mutex.  GM scalar
        // accesses need no local-memory mutex, and print operations are not
        // inspected here by design.
        Value scalarTensor;
        if (auto scalarLoad = dyn_cast<::tla::ScalarLoadOp>(operation))
            scalarTensor = scalarLoad.getSource();
        else if (auto scalarStore = dyn_cast<::tla::ScalarStoreOp>(operation))
            scalarTensor = scalarStore.getDest();
        if (scalarTensor && !operation->getParentOfType<::tla::VecFuncOp>()) {
            auto tensorType = dyn_cast<::tla::TlaTensorType>(scalarTensor.getType());
            if (tensorType && tensorType.getPtr().getAddrspace() == AddressSpace::ub) {
                operation->emitError(
                    "auto_sync='v0' requires UB scalar_load/scalar_store to be "
                    "inside tla.vec.func");
                result = failure();
            }
        }
    });
    return result;
}

using UsedRootsByIdSpace = std::array<DenseSet<Value>, 2>;

static LogicalResult normalizePlanResources(
    SmallVectorImpl<InstructionPlan>& plans, UsedRootsByIdSpace& usedRootsByIdSpace)
{
    for (InstructionPlan& plan : plans) {
        SmallVector<StorageExprPtr, 8> normalized;
        for (const StorageExprPtr& expr : plan.resources) {
            if (!expr || expr->kind == StorageExpr::Kind::Unknown) {
                return plan.op->emitError(
                    "automatic mutex requires every accessed on-chip tensor to resolve "
                    "to a static tla.alloc_ptr capacity/root; bare on-chip addresses "
                    "and changing loop-carried pointers are unsupported");
            }
            if (expr->kind == StorageExpr::Kind::Global)
                continue;
            bool duplicate =
                llvm::any_of(normalized, [&](const StorageExprPtr& other) { return sameStorageExpr(expr, other); });
            if (duplicate)
                continue;
            if (llvm::any_of(
                    normalized, [&](const StorageExprPtr& other) { return haveOverlappingCandidates(expr, other); })) {
                return plan.op->emitError(
                    "automatic mutex found two non-identical resource selections with "
                    "overlapping allocation roots in one instruction");
            }
            normalized.push_back(expr);
            collectCandidateRoots(expr, usedRootsByIdSpace[mutexIdSpaceIndex(plan.idSpace)]);
        }
        plan.resources = std::move(normalized);
    }
    return success();
}

// Dynamic pointer choices may contain conditions defined inside an enclosing
// branch. Recreating the whole condition tree next to a later instruction is
// therefore not dominance-safe. Instead, extend each source scf.if with a
// parallel !tla.mutex result. Branch-local choices are materialized first, and
// the selected mutex follows exactly the same control-flow edges as the pointer
// or tensor it protects.
class ControlFlowMutexMaterializer {
public:
    using MutexesByIdSpace = std::array<DenseMap<Value, Value>, 2>;

    explicit ControlFlowMutexMaterializer(const MutexesByIdSpace& mutexesByIdSpace) : mutexesByIdSpace(mutexesByIdSpace)
    {}

    LogicalResult run(ArrayRef<InstructionPlan> plans)
    {
        for (const InstructionPlan& plan : plans)
            for (const StorageExprPtr& resource : plan.resources)
                collect(resource, plan.idSpace);

        SmallVector<Operation*, 16> selectors;
        selectors.reserve(expressionsByIf.size());
        for (auto& entry : expressionsByIf)
            selectors.push_back(entry.first);
        for (Operation* selector : selectors)
            if (failed(materializeIf(selector)))
                return failure();
        return success();
    }

private:
    struct ScopedStorageExpr {
        StorageExprPtr expr;
        MutexIdSpace idSpace;
    };

    void collect(const StorageExprPtr& expr, MutexIdSpace idSpace)
    {
        if (!expr || expr->kind != StorageExpr::Kind::Select)
            return;
        auto ifOp = expr->selectorSource.getDefiningOp<scf::IfOp>();
        if (ifOp)
            expressionsByIf[ifOp.getOperation()].push_back(ScopedStorageExpr{expr, idSpace});
        collect(expr->thenValue, idSpace);
        collect(expr->elseValue, idSpace);
    }

    FailureOr<Value> getMaterializedMutex(const StorageExprPtr& expr, MutexIdSpace idSpace)
    {
        if (!expr)
            return failure();
        if (expr->kind == StorageExpr::Kind::Root) {
            const DenseMap<Value, Value>& mutexByRoot = mutexesByIdSpace[mutexIdSpaceIndex(idSpace)];
            auto mutex = mutexByRoot.find(expr->root);
            if (mutex == mutexByRoot.end())
                return failure();
            return mutex->second;
        }
        Value materialized = expr->materializedMutexes[mutexIdSpaceIndex(idSpace)];
        if (expr->kind == StorageExpr::Kind::Select && materialized)
            return materialized;
        return failure();
    }

    LogicalResult materializeDependencies(const StorageExprPtr& expr, MutexIdSpace idSpace)
    {
        if (!expr || expr->kind != StorageExpr::Kind::Select)
            return success();
        if (expr->materializedMutexes[mutexIdSpaceIndex(idSpace)])
            return success();
        auto ifOp = expr->selectorSource.getDefiningOp<scf::IfOp>();
        if (!ifOp)
            return failure();
        return materializeIf(ifOp.getOperation());
    }

    LogicalResult materializeIf(Operation* operation)
    {
        if (completed.contains(operation))
            return success();
        if (!visiting.insert(operation).second)
            return operation->emitError("cyclic automatic mutex control-flow provenance");

        auto ifOp = dyn_cast<scf::IfOp>(operation);
        auto group = expressionsByIf.find(operation);
        if (!ifOp || group == expressionsByIf.end())
            return failure();

        DenseMap<uint64_t, unsigned> appendedIndexByResultAndSpace;
        SmallVector<ScopedStorageExpr, 4> canonical;
        SmallVector<std::pair<ScopedStorageExpr, uint64_t>, 8> expressionResultKeys;
        for (const ScopedStorageExpr& scopedExpr : group->second) {
            const StorageExprPtr& expr = scopedExpr.expr;
            auto sourceResult = dyn_cast<OpResult>(expr->selectorSource);
            if (!sourceResult || sourceResult.getOwner() != operation)
                return ifOp.emitError("automatic mutex lost dynamic selector provenance");
            unsigned resultIndex = sourceResult.getResultNumber();
            uint64_t key = (static_cast<uint64_t>(resultIndex) << 1) | mutexIdSpaceIndex(scopedExpr.idSpace);
            expressionResultKeys.emplace_back(scopedExpr, key);
            if (!appendedIndexByResultAndSpace.contains(key)) {
                appendedIndexByResultAndSpace[key] = canonical.size();
                canonical.push_back(scopedExpr);
            } else {
                const StorageExprPtr& previous = canonical[appendedIndexByResultAndSpace[key]].expr;
                if (!sameStorageExpr(previous, expr))
                    return ifOp.emitError(
                        "inconsistent automatic mutex provenance for one scf.if result and core side");
            }
        }

        for (const ScopedStorageExpr& scopedExpr : canonical) {
            const StorageExprPtr& expr = scopedExpr.expr;
            if (failed(materializeDependencies(expr->thenValue, scopedExpr.idSpace)) ||
                failed(materializeDependencies(expr->elseValue, scopedExpr.idSpace)))
                return ifOp.emitError("failed to materialize nested automatic mutex selection");
        }

        scf::YieldOp thenYield = ifOp.thenYield();
        scf::YieldOp elseYield = ifOp.elseYield();
        if (!thenYield || !elseYield)
            return ifOp.emitError("automatic mutex requires a two-branch scf.if selector");

        SmallVector<Value, 8> thenValues(thenYield.getOperands());
        SmallVector<Value, 8> elseValues(elseYield.getOperands());
        SmallVector<Type, 8> resultTypes(ifOp.getResultTypes());
        for (const ScopedStorageExpr& scopedExpr : canonical) {
            const StorageExprPtr& expr = scopedExpr.expr;
            FailureOr<Value> thenMutex = getMaterializedMutex(expr->thenValue, scopedExpr.idSpace);
            FailureOr<Value> elseMutex = getMaterializedMutex(expr->elseValue, scopedExpr.idSpace);
            if (failed(thenMutex) || failed(elseMutex))
                return ifOp.emitError("failed to resolve automatic mutex branch resource");
            thenValues.push_back(*thenMutex);
            elseValues.push_back(*elseMutex);
            resultTypes.push_back(::tla::MutexType::get(ifOp.getContext()));
        }

        unsigned oldResultCount = ifOp.getNumResults();
        OpBuilder builder(ifOp);
        auto newIf = builder.create<scf::IfOp>(
            ifOp.getLoc(), resultTypes, ifOp.getCondition(),
            /*addThenBlock=*/false, /*addElseBlock=*/false);
        newIf->setAttrs(ifOp->getAttrs());
        newIf.getThenRegion().takeBody(ifOp.getThenRegion());
        newIf.getElseRegion().takeBody(ifOp.getElseRegion());

        OpBuilder thenBuilder(thenYield);
        thenBuilder.create<scf::YieldOp>(thenYield.getLoc(), thenValues);
        thenYield.erase();
        OpBuilder elseBuilder(elseYield);
        elseBuilder.create<scf::YieldOp>(elseYield.getLoc(), elseValues);
        elseYield.erase();

        for (unsigned index = 0; index < oldResultCount; ++index)
            ifOp.getResult(index).replaceAllUsesWith(newIf.getResult(index));
        for (auto [scopedExpr, key] : expressionResultKeys) {
            unsigned appendedIndex = appendedIndexByResultAndSpace.lookup(key);
            scopedExpr.expr->materializedMutexes[mutexIdSpaceIndex(scopedExpr.idSpace)] =
                newIf.getResult(oldResultCount + appendedIndex);
        }

        visiting.erase(operation);
        // Keep the original pointer as the stable identity used by the groups
        // collected before any selector is rebuilt. DenseSet lookup does not
        // dereference it after erasure.
        completed.insert(operation);
        ifOp.erase();
        return success();
    }

    const MutexesByIdSpace& mutexesByIdSpace;
    DenseMap<Operation*, SmallVector<ScopedStorageExpr, 4>> expressionsByIf;
    DenseSet<Operation*> visiting;
    DenseSet<Operation*> completed;
};

static FailureOr<Value> materializeMutexSelection(
    const StorageExprPtr& expr, const DenseMap<Value, Value>& mutexByRoot, MutexIdSpace idSpace)
{
    if (!expr)
        return failure();
    if (expr->kind == StorageExpr::Kind::Root) {
        auto mutex = mutexByRoot.find(expr->root);
        if (mutex == mutexByRoot.end())
            return failure();
        return mutex->second;
    }
    Value materialized = expr->materializedMutexes[mutexIdSpaceIndex(idSpace)];
    if (expr->kind == StorageExpr::Kind::Select && materialized)
        return materialized;
    return failure();
}

static std::pair<unsigned, unsigned> mutexIdRange(const StorageExprPtr& expr, const DenseMap<Value, unsigned>& idByRoot)
{
    DenseSet<Value> roots;
    collectCandidateRoots(expr, roots);
    unsigned minId = kMaxAutoMutexIds;
    unsigned maxId = 0;
    for (Value root : roots) {
        auto it = idByRoot.find(root);
        minId = std::min(minId, it->second);
        maxId = std::max(maxId, it->second);
    }
    return {minId, maxId};
}

struct MutexIdAssignment {
    DenseMap<Value, unsigned> idByRoot;
    DenseMap<Value, Value> mutexByRoot;
};

using MutexAssignmentsByIdSpace = std::array<MutexIdAssignment, 2>;

static LogicalResult assignMutexesForIdSpace(
    func::FuncOp func, const TlaScratchAllocationPlan& allocationPlan, const DenseSet<Value>& usedRoots,
    MutexIdSpace idSpace, OpBuilder& declarationBuilder, MutexIdAssignment& assignment)
{
    SmallVector<const TlaScratchAllocation*, 32> used;
    used.reserve(usedRoots.size());
    for (Value root : usedRoots) {
        const TlaScratchAllocation* allocation = allocationPlan.lookup(root);
        if (!allocation)
            return func.emitError("automatic mutex lost scratch allocation provenance");
        used.push_back(allocation);
    }
    llvm::sort(used, [](const TlaScratchAllocation* lhs, const TlaScratchAllocation* rhs) {
        unsigned lhsRank = addressSpaceRank(lhs->addressSpace);
        unsigned rhsRank = addressSpaceRank(rhs->addressSpace);
        return std::tie(lhsRank, lhs->base) < std::tie(rhsRank, rhs->base);
    });
    if (used.size() > kMaxAutoMutexIds)
        return func.emitError() << "automatic mutex requires " << used.size() << " IDs for the "
                                << stringifyMutexIdSpace(idSpace) << " side, exceeding the per-core hardware limit of "
                                << kMaxAutoMutexIds;

    for (auto [index, lhs] : llvm::enumerate(used)) {
        for (const TlaScratchAllocation* rhs : llvm::drop_begin(used, index + 1)) {
            if (lhs->addressSpace != rhs->addressSpace)
                continue;
            if (lhs->base < rhs->end && rhs->base < lhs->end)
                return func.emitError() << "automatic mutex cannot assign distinct IDs to overlapping "
                                           "scratch intervals in "
                                        << stringifyAddressSpace(lhs->addressSpace) << ": [" << lhs->base << ", "
                                        << lhs->end << ") and [" << rhs->base << ", " << rhs->end << ")";
        }
    }

    for (auto [id, allocation] : llvm::enumerate(used)) {
        assignment.idByRoot[allocation->root] = id;
        std::string resource;
        llvm::raw_string_ostream stream(resource);
        stream << "auto_" << stringifyAddressSpace(allocation->addressSpace) << "_" << allocation->base << "_"
               << allocation->sizeBytes;
        auto mutex = declarationBuilder.create<::tla::MutexOp>(
            func.getLoc(), ::tla::MutexType::get(func.getContext()), resource, id);
        assignment.mutexByRoot[allocation->root] = mutex.getMutex();
    }
    return success();
}

static LogicalResult insertMutexesForFunction(
    func::FuncOp func, const TlaScratchAllocationPlan& allocationPlan, SmallVectorImpl<InstructionPlan>& plans,
    const UsedRootsByIdSpace& usedRootsByIdSpace)
{
    MutexAssignmentsByIdSpace assignments;
    OpBuilder declarationBuilder = OpBuilder::atBlockBegin(&func.getBody().front());
    for (MutexIdSpace idSpace : {MutexIdSpace::Cube, MutexIdSpace::Vector}) {
        unsigned index = mutexIdSpaceIndex(idSpace);
        if (failed(assignMutexesForIdSpace(
                func, allocationPlan, usedRootsByIdSpace[index], idSpace, declarationBuilder, assignments[index])))
            return failure();
    }

    ControlFlowMutexMaterializer::MutexesByIdSpace mutexesByIdSpace;
    for (MutexIdSpace idSpace : {MutexIdSpace::Cube, MutexIdSpace::Vector}) {
        unsigned index = mutexIdSpaceIndex(idSpace);
        mutexesByIdSpace[index] = assignments[index].mutexByRoot;
    }
    ControlFlowMutexMaterializer controlFlowMaterializer(mutexesByIdSpace);
    if (failed(controlFlowMaterializer.run(plans)))
        return func.emitError("failed to carry automatic mutexes through dynamic pointer control flow");

    for (InstructionPlan& plan : plans) {
        const MutexIdAssignment& assignment = assignments[mutexIdSpaceIndex(plan.idSpace)];
        llvm::sort(plan.resources, [&](const StorageExprPtr& lhs, const StorageExprPtr& rhs) {
            return mutexIdRange(lhs, assignment.idByRoot).first < mutexIdRange(rhs, assignment.idByRoot).first;
        });
        for (auto pair : llvm::zip(plan.resources, llvm::drop_begin(plan.resources))) {
            auto previous = mutexIdRange(std::get<0>(pair), assignment.idByRoot);
            auto next = mutexIdRange(std::get<1>(pair), assignment.idByRoot);
            if (previous.second >= next.first)
                return plan.op->emitError(
                    "automatic mutex cannot prove a stable stack lock order for "
                    "interleaved dynamic resource selections");
        }

        OpBuilder before(plan.op);
        SmallVector<Value, 8> selectedMutexes;
        selectedMutexes.reserve(plan.resources.size());
        for (const StorageExprPtr& resource : plan.resources) {
            FailureOr<Value> mutex = materializeMutexSelection(resource, assignment.mutexByRoot, plan.idSpace);
            if (failed(mutex))
                return plan.op->emitError("failed to materialize dynamic automatic mutex selection");
            selectedMutexes.push_back(*mutex);
        }
        ::tla::PipeAttr pipe = ::tla::PipeAttr::get(func.getContext(), plan.pipe);
        for (Value mutex : selectedMutexes)
            before.create<::tla::MutexLockOp>(plan.op->getLoc(), mutex, pipe);

        OpBuilder after(plan.op);
        after.setInsertionPointAfter(plan.op);
        for (Value mutex : llvm::reverse(selectedMutexes))
            after.create<::tla::MutexUnlockOp>(plan.op->getLoc(), mutex, pipe);
    }
    return success();
}

class TlaInsertAutoMutexPass : public PassWrapper<TlaInsertAutoMutexPass, OperationPass<ModuleOp>> {
public:
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TlaInsertAutoMutexPass)

    StringRef getArgument() const override
    {
        return "tla-insert-auto-mutex";
    }
    StringRef getName() const override
    {
        return "TlaInsertAutoMutexPass";
    }
    StringRef getDescription() const override
    {
        return "Insert instruction-scoped local mutex synchronization from static "
               "TLA scratch allocation provenance";
    }

    void getDependentDialects(DialectRegistry& registry) const override
    {
        registry.insert<arith::ArithDialect, func::FuncDialect, scf::SCFDialect, ::tla::TlaDialect>();
    }

    void runOnOperation() override
    {
        ModuleOp module = getOperation();
        FailureOr<TlaScratchAllocationPlan> allocationPlan = planTlaScratchAllocations(module);
        if (failed(allocationPlan)) {
            signalPassFailure();
            return;
        }

        for (func::FuncOp func : module.getOps<func::FuncOp>()) {
            auto version = func->getAttrOfType<StringAttr>(kAutoSyncAttrName);
            if (!version)
                continue;
            if (version.getValue() != "v0") {
                func.emitError() << "unsupported tla.auto_sync version '" << version.getValue() << "'; expected 'v0'";
                signalPassFailure();
                return;
            }
            if (func.isDeclaration()) {
                func->removeAttr(kAutoSyncAttrName);
                continue;
            }
            if (failed(validateNoExternCalls(func)) || failed(validateNoManualLocalSync(func)) ||
                failed(validateCopyUnitFlags(func))) {
                signalPassFailure();
                return;
            }

            StorageResolver resolver(*allocationPlan);
            SmallVector<InstructionPlan, 32> plans;
            if (failed(collectInstructionPlans(func, resolver, plans))) {
                signalPassFailure();
                return;
            }
            UsedRootsByIdSpace usedRootsByIdSpace;
            if (failed(normalizePlanResources(plans, usedRootsByIdSpace)) ||
                failed(insertMutexesForFunction(func, *allocationPlan, plans, usedRootsByIdSpace))) {
                signalPassFailure();
                return;
            }
            // This is a compile-time control attribute, not backend ABI metadata.
            func->removeAttr(kAutoSyncAttrName);
        }
    }
};

} // namespace

std::unique_ptr<Pass> createTlaInsertAutoMutexPass()
{
    return std::make_unique<TlaInsertAutoMutexPass>();
}

void registerTlaInsertAutoMutexPass()
{
    PassRegistration<TlaInsertAutoMutexPass>();
}

} // namespace tla
