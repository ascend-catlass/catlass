#include "PassesCommon.h"
#include "PassesInternal.h"

#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Builders.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"

// tla-unalign-ureg: Python ``tile.load`` / ``tile.store`` UnAlign UPDATE omit
// ureg; this pass fills each UPDATE's ureg input and inserts Pre / Post.
//
//   1) Map each window ``!tla.tensor`` **per direction** (load vs store) to the
//      ``!tla.ureg`` that the next same-direction UPDATE should take. After an
//      UPDATE, the next window rebuilt from ``next_ptr`` gets this UPDATE's
//      ureg result on that direction only (store→load on the same window must
//      not reuse the store ureg).
//   2) If the consumed tensor has no ureg yet and is not a clean for iter_arg:
//      insert ``load_unalign_pre`` / ``create_ureg`` as the previous op of
//      that UPDATE in the same block (per-trip dual-instr). This is also the
//      path for ``for`` ⊃ ``if`` ⊃ UPDATE (UPDATE is not a direct for-body op,
//      so (3) never rewrites it).
//   3) If the consumed tensor is already a *clean* ``scf.for`` iter_arg (no
//      intervening ``scf.if`` between that for and UPDATE): rewrite that for
//      to also carry a ureg; Pre / create_ureg before the for as ureg init;
//      yield the ureg of the UPDATE whose next window is the yielded tensor
//      (a same-direction chain of UPDATEs in one trip is wired end-to-end).
//      Conflict model (sound, same-base only): peel tensors/ptrs to a
//      ``storageRoot`` (allocate / stable base). Same root ⇒ May/Must conflict
//      for RAW/WAR/WAW; different root ⇒ no conflict. ``ptr_add`` / ``recast_ptr``
//      peel through. Recognized mem ops: load/store (+UnAlign/Align UPDATE),
//      ``tla.scalar_load`` / ``tla.scalar_store``, ``tla.copy`` (src/dst),
//      ``tla.fill`` (dst). No user-addi heuristic.
//      Store/load for-elim is skipped when the for body has another mem op on
//      the same root (excluding this UPDATE's own same-direction chain).
//      ``before=nullptr`` scans the whole for body.
//      Supported elim nestings: plain for; if⊃for (any depth); for⊃for (any
//      depth, innermost clean for).
//      Cross-for 续流 is intentionally unsupported: prologue ureg is never
//      wired as for init, and a for-carried ureg result is never reused by a
//      later UPDATE (epilogue / next for). Those sites dual-instr with a fresh
//      Pre/create_ureg; Post of the for stream sits after the for.
//   4) Store: insert ``store_unalign_post`` after the ureg that is not taken
//      as input by a later store UPDATE. Tip sits after a for when the ureg
//      is a yielded for result (elim flush). Multiple Posts after the same
//      for are appended in walk order (尾插). Straight-line store→store reuse
//      is cut when an intervening same-root load/store conflicts.
// Unsupported: if/else merge of UPDATE tiles, or two UnAlign UPDATEs on the
// same for-carried window without a unique back-edge ureg → emitError.

namespace tla {
namespace {

static constexpr StringLiteral kPassDiagPrefix = "tla-unalign-ureg: ";

/// Load- vs store-stream ureg tips keyed by window tensor.
struct UregMaps {
    DenseMap<Value, Value> load;
    DenseMap<Value, Value> store;

    DenseMap<Value, Value>& forDir(bool isLoad)
    {
        return isLoad ? load : store;
    }

    void clear()
    {
        load.clear();
        store.clear();
    }
};

/// Next ``!tla.tensor`` window rebuilt from UPDATE ``next_ptr`` (make_tensor*).
static Value nextWindowTensorFromUpdate(Operation* updateOp)
{
    Value nextPtr;
    if (auto load = dyn_cast<::tla::LoadUnalignUpdateOp>(updateOp))
        nextPtr = load.getNextPtr();
    else if (auto store = dyn_cast<::tla::StoreUnalignUpdateOp>(updateOp))
        nextPtr = store.getNextPtr();
    else
        return Value();
    for (Operation* user : nextPtr.getUsers()) {
        if (auto make = dyn_cast<::tla::MakeTensorOp>(user)) {
            if (make.getPtr() == nextPtr)
                return make.getResult();
        }
        if (auto makeLike = dyn_cast<::tla::MakeTensorLikeOp>(user)) {
            if (makeLike.getPtr() == nextPtr)
                return makeLike.getResult();
        }
    }
    return Value();
}

static Value uregOutOf(Operation* updateOp)
{
    if (auto load = dyn_cast<::tla::LoadUnalignUpdateOp>(updateOp))
        return load.getUregOut();
    if (auto store = dyn_cast<::tla::StoreUnalignUpdateOp>(updateOp))
        return store.getUregOut();
    return Value();
}

static Value windowTensorOf(Operation* updateOp, bool isLoad)
{
    if (isLoad) {
        if (auto load = dyn_cast<::tla::LoadUnalignUpdateOp>(updateOp))
            return load.getSource();
    } else if (auto store = dyn_cast<::tla::StoreUnalignUpdateOp>(updateOp)) {
        return store.getDest();
    }
    return Value();
}

/// Peel trivial casts, then look up the ureg for ``tensor`` on one direction.
static Value lookupUregForTensor(Value tensor, DenseMap<Value, Value>& uregForTensor)
{
    Value v = tensor;
    DenseSet<Value> seen;
    while (seen.insert(v).second) {
        if (Value ureg = uregForTensor.lookup(v))
            return ureg;
        Operation* def = v.getDefiningOp();
        if (def && isa<UnrealizedConversionCastOp>(def) && def->getNumOperands() == 1) {
            v = def->getOperand(0);
            continue;
        }
        break;
    }
    return Value();
}

static std::optional<unsigned> matchIterArgIndex(Value tensor, scf::ForOp forOp)
{
    Value v = tensor;
    DenseSet<Value> seen;
    while (seen.insert(v).second) {
        if (auto ba = dyn_cast<BlockArgument>(v)) {
            if (ba.getOwner() == forOp.getBody() && ba.getArgNumber() >= 1)
                return ba.getArgNumber() - 1;
            return std::nullopt;
        }
        Operation* def = v.getDefiningOp();
        if (!def || isa<::tla::MakeTensorOp>(def))
            return std::nullopt;
        if (isa<UnrealizedConversionCastOp>(def) && def->getNumOperands() == 1) {
            v = def->getOperand(0);
            continue;
        }
        return std::nullopt;
    }
    return std::nullopt;
}

/// True when ``updateOp`` sits under ``forOp`` with no intervening ``scf.if``.
/// Nested ``scf.for`` is allowed; ``for`` ⊃ ``if`` ⊃ UPDATE is not (dual-instr).
static bool isCleanForUregCarry(Operation* updateOp, scf::ForOp forOp)
{
    Operation* parent = updateOp->getParentOp();
    while (parent && parent != forOp.getOperation()) {
        if (isa<scf::IfOp>(parent))
            return false;
        parent = parent->getParentOp();
    }
    return parent == forOp.getOperation();
}

/// Peel derived tensor/ptr constructors to a stable UB/base identity.
/// Same root ⇒ same underlying storage (conflict for RAW/WAR/WAW).
///
/// Closed peel list (extend when a new derived constructor is added):
///   tile_view | make_tensor | make_tensor_like | tensor_ptr | ptr_add
///   | recast_ptr | UnAlign/Align UPDATE next_ptr → window | ptrtoint/inttoptr
///   | scf.for iter_arg → matching init | 1-op unrealized_conversion_cast
static Value storageRoot(Value v)
{
    DenseSet<Value> seen;
    while (v && seen.insert(v).second) {
        if (auto ba = dyn_cast<BlockArgument>(v)) {
            if (auto forOp = dyn_cast<scf::ForOp>(ba.getOwner()->getParentOp())) {
                unsigned n = ba.getArgNumber();
                if (n >= 1 && n - 1 < forOp.getInitArgs().size()) {
                    v = forOp.getInitArgs()[n - 1];
                    continue;
                }
            }
            break;
        }
        Operation* def = v.getDefiningOp();
        if (!def)
            break;
        if (isa<UnrealizedConversionCastOp>(def) && def->getNumOperands() == 1) {
            v = def->getOperand(0);
            continue;
        }
        if (auto tile = dyn_cast<::tla::TileViewOp>(def)) {
            v = tile.getSource();
            continue;
        }
        if (auto make = dyn_cast<::tla::MakeTensorOp>(def)) {
            v = make.getPtr();
            continue;
        }
        if (auto makeLike = dyn_cast<::tla::MakeTensorLikeOp>(def)) {
            v = makeLike.getPtr();
            continue;
        }
        if (auto tptr = dyn_cast<::tla::TensorPtrOp>(def)) {
            v = tptr.getSrc();
            continue;
        }
        if (auto padd = dyn_cast<::tla::PtrAddOp>(def)) {
            v = padd.getPtr();
            continue;
        }
        if (auto recast = dyn_cast<::tla::RecastPtrOp>(def)) {
            v = recast.getSrc();
            continue;
        }
        // UPDATE next_ptr is the same storage root as the window it advanced.
        if (auto loadU = dyn_cast<::tla::LoadUnalignUpdateOp>(def)) {
            v = loadU.getSource();
            continue;
        }
        if (auto storeU = dyn_cast<::tla::StoreUnalignUpdateOp>(def)) {
            v = storeU.getDest();
            continue;
        }
        if (auto loadU = dyn_cast<::tla::LoadUpdateOp>(def)) {
            v = loadU.getSource();
            continue;
        }
        if (auto storeU = dyn_cast<::tla::StoreUpdateOp>(def)) {
            v = storeU.getDest();
            continue;
        }
        if (auto pti = dyn_cast<::tla::PtrToIntOp>(def)) {
            v = pti.getPtr();
            continue;
        }
        if (auto itp = dyn_cast<::tla::IntToPtrOp>(def)) {
            v = itp.getOperand();
            continue;
        }
        if (def->getName().getStringRef() == "llvm.ptrtoint") {
            v = def->getOperand(0);
            continue;
        }
        // Constant byte/element adjust on an integerized ptr: keep peeling the
        // ptrtoint side when obvious; otherwise stop (unknown → distinct SSA).
        if (auto addi = dyn_cast<arith::AddIOp>(def)) {
            Value lhs = addi.getLhs();
            Value rhs = addi.getRhs();
            if (lhs.getDefiningOp<::tla::PtrToIntOp>() ||
                (lhs.getDefiningOp() && lhs.getDefiningOp()->getName().getStringRef() == "llvm.ptrtoint")) {
                v = lhs;
                continue;
            }
            if (rhs.getDefiningOp<::tla::PtrToIntOp>() ||
                (rhs.getDefiningOp() && rhs.getDefiningOp()->getName().getStringRef() == "llvm.ptrtoint")) {
                v = rhs;
                continue;
            }
            break;
        }
        break;
    }
    return v;
}

static bool sharesRootWithWindows(Value accessWindow, Value windowA, Value windowB)
{
    if (!accessWindow)
        return false;
    Value root = storageRoot(accessWindow);
    if (!root)
        return false;
    if (windowA && storageRoot(windowA) == root)
        return true;
    if (windowB && storageRoot(windowB) == root)
        return true;
    return false;
}

static Value loadSourceOf(Operation* op)
{
    if (auto load = dyn_cast<::tla::LoadOp>(op))
        return load.getSource();
    if (auto load = dyn_cast<::tla::LoadUpdateOp>(op))
        return load.getSource();
    if (auto load = dyn_cast<::tla::LoadUnalignUpdateOp>(op))
        return load.getSource();
    if (auto load = dyn_cast<::tla::ScalarLoadOp>(op))
        return load.getSource();
    return Value();
}

static Value storeDestOf(Operation* op)
{
    if (auto store = dyn_cast<::tla::StoreOp>(op))
        return store.getDest();
    if (auto store = dyn_cast<::tla::StoreUpdateOp>(op))
        return store.getDest();
    if (auto store = dyn_cast<::tla::StoreUnalignUpdateOp>(op))
        return store.getDest();
    if (auto store = dyn_cast<::tla::ScalarStoreOp>(op))
        return store.getDest();
    return Value();
}

/// Load touches the same storage root as ``windowA`` / ``windowB``.
static bool loadConflictsWindows(Operation* op, Value windowA, Value windowB)
{
    return sharesRootWithWindows(loadSourceOf(op), windowA, windowB);
}

/// Store touches the same storage root as ``windowA`` / ``windowB``.
static bool storeConflictsWindows(Operation* op, Value windowA, Value windowB)
{
    return sharesRootWithWindows(storeDestOf(op), windowA, windowB);
}

/// ``tla.copy`` / ``tla.fill`` may touch the same root (copy: src or dst).
static bool otherMemConflictsWindows(Operation* op, Value windowA, Value windowB)
{
    if (auto copy = dyn_cast<::tla::CopyOp>(op)) {
        return sharesRootWithWindows(copy.getDst(), windowA, windowB) ||
               sharesRootWithWindows(copy.getSrc(), windowA, windowB);
    }
    if (auto fill = dyn_cast<::tla::FillOp>(op))
        return sharesRootWithWindows(fill.getDst(), windowA, windowB);
    return false;
}

/// Any recognized mem op that shares a storage root with the windows.
static bool memOpConflictsWindows(Operation* op, Value windowA, Value windowB)
{
    return loadConflictsWindows(op, windowA, windowB) || storeConflictsWindows(op, windowA, windowB) ||
           otherMemConflictsWindows(op, windowA, windowB);
}

/// Strict program-order ``a`` before ``b`` under structured SCF nesting.
/// Same block ⇒ ``isBeforeInBlock``; otherwise climb to a common block (so a
/// load inside ``scf.if`` / ``scf.for`` between two parent-block stores is
/// ordered correctly). ``a`` ancestor-of ``b`` ⇒ ``a`` starts before ``b``.
static bool isStrictlyBefore(Operation* a, Operation* b)
{
    if (!a || !b || a == b)
        return false;
    if (a->getBlock() == b->getBlock())
        return a->isBeforeInBlock(b);
    if (b->isProperAncestor(a))
        return false;
    if (a->isProperAncestor(b))
        return true;

    SmallVector<Operation*, 8> aChain;
    for (Operation* x = a; x; x = x->getParentOp())
        aChain.push_back(x);
    for (Operation* y = b; y; y = y->getParentOp()) {
        for (Operation* x : aChain) {
            if (x->getBlock() == y->getBlock())
                return x->isBeforeInBlock(y);
        }
    }
    return false;
}

/// ``op`` executes after ``after`` and, if ``before`` is set, before ``before``.
/// Nested region ops are ordered via ancestors in a common block.
static bool isBetweenOps(Operation* op, Operation* after, Operation* before)
{
    if (!isStrictlyBefore(after, op))
        return false;
    if (before && !isStrictlyBefore(op, before))
        return false;
    return true;
}

/// Walk ``region`` for a matching op. Skips ``after``. If ``before`` is set,
/// require ``after ≺ op ≺ before`` (nested SCF ok); otherwise any op in the
/// region counts (for-elim when Post would sit after the for).
template <typename MatchFn>
static bool walkRegionForMatch(Region& region, Operation* after, Operation* before, MatchFn match)
{
    bool found = false;
    region.walk([&](Operation* op) {
        if (found || op == after)
            return;
        if (before && !isBetweenOps(op, after, before))
            return;
        if (match(op))
            found = true;
    });
    return found;
}

/// RAW/WAR/WAW (+ copy/fill): any recognized mem op on the same storage root.
static bool hasConflictingMemOp(Region& region, Operation* after, Operation* before, Value windowA, Value windowB)
{
    return walkRegionForMatch(
        region, after, before, [&](Operation* op) { return memOpConflictsWindows(op, windowA, windowB); });
}

/// Same-direction UnAlign UPDATEs reachable by following next windows from ``head``
/// inside ``forOp``'s direct body (own stream chain; excluded from elim conflicts).
static DenseSet<Operation*> ownSameDirUpdateChain(scf::ForOp forOp, Operation* head, bool isLoad)
{
    DenseSet<Operation*> chain;
    Operation* cur = head;
    while (cur && chain.insert(cur).second) {
        Value next = nextWindowTensorFromUpdate(cur);
        if (!next)
            break;
        Operation* nextUpdate = nullptr;
        unsigned hits = 0;
        for (Operation& op : *forOp.getBody()) {
            if (isLoad) {
                auto load = dyn_cast<::tla::LoadUnalignUpdateOp>(&op);
                if (!load || load.getSource() != next)
                    continue;
            } else {
                auto store = dyn_cast<::tla::StoreUnalignUpdateOp>(&op);
                if (!store || store.getDest() != next)
                    continue;
            }
            nextUpdate = &op;
            ++hits;
        }
        if (hits != 1)
            break;
        cur = nextUpdate;
    }
    return chain;
}

/// Store for-carry puts Post after the for. Skip when the body has any other
/// same-root mem op (load or store), excluding this store's own UPDATE chain.
static bool forBodyBlocksStoreUregElim(scf::ForOp forOp, Operation* storeUpdate)
{
    Value storeDest = cast<::tla::StoreUnalignUpdateOp>(storeUpdate).getDest();
    Value storeNext = nextWindowTensorFromUpdate(storeUpdate);
    DenseSet<Operation*> own = ownSameDirUpdateChain(forOp, storeUpdate, /*isLoad=*/false);
    return walkRegionForMatch(forOp.getRegion(), storeUpdate, /*before=*/nullptr, [&](Operation* op) {
        if (own.contains(op))
            return false;
        return memOpConflictsWindows(op, storeDest, storeNext);
    });
}

/// Load for-carry skips elim when the body has another same-root mem op
/// (excluding this load's own UPDATE chain).
static bool forBodyBlocksLoadUregElim(scf::ForOp forOp, Operation* loadUpdate)
{
    Value loadSrc = cast<::tla::LoadUnalignUpdateOp>(loadUpdate).getSource();
    Value loadNext = nextWindowTensorFromUpdate(loadUpdate);
    DenseSet<Operation*> own = ownSameDirUpdateChain(forOp, loadUpdate, /*isLoad=*/true);
    return walkRegionForMatch(forOp.getRegion(), loadUpdate, /*before=*/nullptr, [&](Operation* op) {
        if (own.contains(op))
            return false;
        return memOpConflictsWindows(op, loadSrc, loadNext);
    });
}

/// Prior store ``ureg_out`` must not feed a later store across a same-root conflict.
static bool storeUregReuseCrossesConflict(Value ureg, Operation* consumerStore)
{
    auto prevStore = ureg.getDefiningOp<::tla::StoreUnalignUpdateOp>();
    if (!prevStore || prevStore.getUregOut() != ureg)
        return false;
    Value dest = prevStore.getDest();
    Value next = nextWindowTensorFromUpdate(prevStore.getOperation());
    return hasConflictingMemOp(
        *prevStore->getBlock()->getParent(), prevStore.getOperation(), consumerStore, dest, next);
}

/// Prior load ``ureg_out`` must not feed a later load across a same-root conflict.
static bool loadUregReuseCrossesConflict(Value ureg, Operation* consumerLoad, Value loadWindow)
{
    auto prevLoad = ureg.getDefiningOp<::tla::LoadUnalignUpdateOp>();
    if (!prevLoad || prevLoad.getUregOut() != ureg)
        return false;
    Value loadNext = nextWindowTensorFromUpdate(prevLoad.getOperation());
    return hasConflictingMemOp(
        *prevLoad->getBlock()->getParent(), prevLoad.getOperation(), consumerLoad, loadWindow, loadNext);
}

/// Same-direction UPDATEs in ``forOp``'s direct body whose window is ``window``.
static SmallVector<Operation*, 2> findUpdatesOnWindow(scf::ForOp forOp, Value window, bool isLoad)
{
    SmallVector<Operation*, 2> found;
    for (Operation& op : *forOp.getBody()) {
        if (isLoad) {
            auto load = dyn_cast<::tla::LoadUnalignUpdateOp>(&op);
            if (!load || load.getSource() != window)
                continue;
        } else {
            auto store = dyn_cast<::tla::StoreUnalignUpdateOp>(&op);
            if (!store || store.getDest() != window)
                continue;
        }
        found.push_back(&op);
    }
    return found;
}

/// Build the unique same-direction UPDATE chain from ``head`` (on iter_arg) to
/// the yielded next-window tensor. Empty ⇒ cannot eliminate (fall back dual).
/// Failure ⇒ truly ambiguous (emitError).
static FailureOr<SmallVector<Operation*, 4>> buildUregCarryChain(
    scf::ForOp forOp, unsigned tensorIterIdx, bool isLoad, Operation* head)
{
    auto yield = cast<scf::YieldOp>(forOp.getBody()->getTerminator());
    if (tensorIterIdx >= yield.getNumOperands())
        return head->emitError(Twine(kPassDiagPrefix) + "yield missing tensor iter_arg operand"), failure();
    Value yieldTensor = yield.getOperand(tensorIterIdx);

    SmallVector<Operation*, 4> chain;
    Operation* cur = head;
    DenseSet<Operation*> seen;
    while (cur && seen.insert(cur).second) {
        chain.push_back(cur);
        Value next = nextWindowTensorFromUpdate(cur);
        if (!next)
            return SmallVector<Operation*, 4>{};
        if (next == yieldTensor)
            return chain;
        SmallVector<Operation*, 2> nextUpdates = findUpdatesOnWindow(forOp, next, isLoad);
        if (nextUpdates.size() > 1) {
            return cur->emitError(
                       Twine(kPassDiagPrefix) +
                       "multiple UnAlign UPDATE ops on the same window "
                       "without a unique back-edge ureg SSA"),
                   failure();
        }
        if (nextUpdates.empty())
            return SmallVector<Operation*, 4>{};
        cur = nextUpdates.front();
    }
    return SmallVector<Operation*, 4>{};
}

static void assignUregIn(Operation* updateOp, bool isLoad, Value uregIn)
{
    if (isLoad)
        cast<::tla::LoadUnalignUpdateOp>(updateOp).getUregInMutable().assign(uregIn);
    else
        cast<::tla::StoreUnalignUpdateOp>(updateOp).getUregInMutable().assign(uregIn);
}

/// Drop Pre/create_ureg that only fed an UPDATE we are about to rewire.
static void eraseUnusedUregInit(Value uregIn)
{
    if (!uregIn)
        return;
    Operation* def = uregIn.getDefiningOp();
    if (!def)
        return;
    if (!isa<::tla::LoadUnalignPreOp, ::tla::CreateUregOp>(def))
        return;
    if (!uregIn.use_empty())
        return;
    def->erase();
}

/// True when ``window`` is an ``scf.for`` result whose body still has a deferred
/// clean same-direction UPDATE on that iter — after for-carry rewrite the map
/// will bind ``for.result → ureg``; filling now would wrongly open a new stream.
static bool forResultAwaitsUregCarry(Value window, bool isLoad)
{
    auto forOp = dyn_cast_or_null<scf::ForOp>(window.getDefiningOp());
    if (!forOp)
        return false;
    std::optional<unsigned> resIdx;
    for (auto [i, res] : llvm::enumerate(forOp.getResults())) {
        if (res == window) {
            resIdx = i;
            break;
        }
    }
    if (!resIdx || *resIdx >= forOp.getNumRegionIterArgs())
        return false;

    for (Operation& op : *forOp.getBody()) {
        Operation* update = nullptr;
        Value windowInBody;
        if (isLoad) {
            auto load = dyn_cast<::tla::LoadUnalignUpdateOp>(&op);
            if (!load || load.getUregIn())
                continue;
            windowInBody = load.getSource();
            update = &op;
        } else {
            auto store = dyn_cast<::tla::StoreUnalignUpdateOp>(&op);
            if (!store || store.getUregIn())
                continue;
            windowInBody = store.getDest();
            update = &op;
        }
        auto idx = matchIterArgIndex(windowInBody, forOp);
        if (!idx || *idx != *resIdx || !isCleanForUregCarry(update, forOp))
            continue;
        bool blocked = isLoad ? forBodyBlocksLoadUregElim(forOp, update) : forBodyBlocksStoreUregElim(forOp, update);
        if (!blocked)
            return true;
    }
    return false;
}

/// True when ``ureg`` is an ``scf.for`` result (for-carried stream tip).
static bool isScfForResultUreg(Value ureg)
{
    return ureg && isa_and_nonnull<scf::ForOp>(ureg.getDefiningOp());
}

/// Ureg input for ``updateOp``: reuse the per-direction map, defer if window is
/// a *clean* for iter_arg eligible for elim (or a for result awaiting that
/// carry), or insert Pre / create_ureg.
///
/// For-carried ureg results are never reused past the for (no cross-for 续流):
/// epilogue / next-for UPDATEs open a fresh Pre/create_ureg stream.
static Value bindOrInsertUregIn(Operation* updateOp, Value windowTensor, bool isLoad, UregMaps& uregMaps)
{
    if (Value ureg = lookupUregForTensor(windowTensor, uregMaps.forDir(isLoad))) {
        // Do not continue a for-carried ureg into ring-out / next-for UPDATEs.
        if (!isScfForResultUreg(ureg)) {
            // Same-root intervening load/store (RAW/WAR/WAW) cuts ureg reuse.
            bool reuseOk = isLoad ? !loadUregReuseCrossesConflict(ureg, updateOp, windowTensor) :
                                    !storeUregReuseCrossesConflict(ureg, updateOp);
            if (reuseOk)
                return ureg;
        }
    }

    // Innermost enclosing for that carries ``windowTensor``: defer only when
    // clean and elim is not blocked. ``for`` ⊃ ``if`` ⊃ UPDATE is unclean
    // → fall through to dual-instr Pre/create_ureg inside the if.
    if (auto forOp = updateOp->getParentOfType<scf::ForOp>()) {
        if (matchIterArgIndex(windowTensor, forOp) && isCleanForUregCarry(updateOp, forOp)) {
            bool blocked =
                isLoad ? forBodyBlocksLoadUregElim(forOp, updateOp) : forBodyBlocksStoreUregElim(forOp, updateOp);
            if (!blocked)
                return Value();
        }
    }

    // After the for: wait for for-carry rewrite so Post can tip after the for;
    // do not open a second stream on the window while carry is still pending.
    // Once rewritten, the for result is intentionally *not* mapped to ureg for
    // reuse — epilogue dual-instrs (see no cross-for 续流).
    if (forResultAwaitsUregCarry(windowTensor, isLoad))
        return Value();

    OpBuilder b(updateOp);
    Type uregTy = ::tla::UregType::get(updateOp->getContext());
    if (isLoad)
        return b.create<::tla::LoadUnalignPreOp>(updateOp->getLoc(), uregTy, windowTensor).getResult();
    return b.create<::tla::CreateUregOp>(updateOp->getLoc(), uregTy).getResult();
}

static void fillUregInBlock(Block& block, UregMaps& uregMaps)
{
    for (Operation& op : block) {
        Value windowTensor;
        bool isLoad = true;
        if (auto load = dyn_cast<::tla::LoadUnalignUpdateOp>(&op)) {
            if (load.getUregIn()) {
                if (Value next = nextWindowTensorFromUpdate(&op))
                    uregMaps.load[next] = load.getUregOut();
                continue;
            }
            windowTensor = load.getSource();
            isLoad = true;
        } else if (auto store = dyn_cast<::tla::StoreUnalignUpdateOp>(&op)) {
            if (store.getUregIn()) {
                if (Value next = nextWindowTensorFromUpdate(&op))
                    uregMaps.store[next] = store.getUregOut();
                continue;
            }
            windowTensor = store.getDest();
            isLoad = false;
        } else {
            continue;
        }

        Value uregIn = bindOrInsertUregIn(&op, windowTensor, isLoad, uregMaps);
        if (!uregIn)
            continue; // deferred to for-carry
        assignUregIn(&op, isLoad, uregIn);
        if (Value next = nextWindowTensorFromUpdate(&op))
            uregMaps.forDir(isLoad)[next] = uregOutOf(&op);
    }
}

static LogicalResult addUregCarryForChain(
    scf::ForOp forOp, unsigned tensorIterIdx, ArrayRef<Operation*> chain, Value uregInit, bool isLoad)
{
    if (chain.empty()) {
        return forOp.emitError(Twine(kPassDiagPrefix) + "empty UnAlign UPDATE chain for ureg carry");
    }
    Location loc = forOp.getLoc();
    OpBuilder b(forOp);
    SmallVector<Value> newInits(forOp.getInitArgs().begin(), forOp.getInitArgs().end());
    newInits.push_back(uregInit);

    scf::ForOp newFor =
        b.create<scf::ForOp>(loc, forOp.getLowerBound(), forOp.getUpperBound(), forOp.getStep(), newInits);

    IRMapping mapping;
    mapping.map(forOp.getInductionVar(), newFor.getInductionVar());
    auto newIterArgs = newFor.getRegionIterArgs();
    for (auto [oldArg, newArg] : llvm::zip(forOp.getRegionIterArgs(), newIterArgs.drop_back()))
        mapping.map(oldArg, newArg);
    BlockArgument newUregArg = newIterArgs.back();

    Block& oldBody = *forOp.getBody();
    Block& newBody = *newFor.getBody();
    if (!newBody.empty())
        newBody.getTerminator()->erase();
    OpBuilder bodyBuilder = OpBuilder::atBlockEnd(&newBody);

    DenseMap<Operation*, Operation*> clonedByOriginal;
    for (Operation& op : llvm::drop_end(oldBody.getOperations())) {
        Operation* cloned = bodyBuilder.clone(op, mapping);
        clonedByOriginal[&op] = cloned;
    }

    SmallVector<Operation*, 4> clonedChain;
    clonedChain.reserve(chain.size());
    for (Operation* original : chain) {
        Operation* cloned = clonedByOriginal.lookup(original);
        if (!cloned) {
            return original->emitError(Twine(kPassDiagPrefix) + "failed to clone UPDATE into rewritten for");
        }
        clonedChain.push_back(cloned);
    }

    // Rewire the whole same-direction chain onto the for-carried ureg; drop any
    // per-trip Pre/create_ureg that fillUregInBlock may have attached to tails.
    Value prevUreg = newUregArg;
    for (Operation* cloned : clonedChain) {
        Value oldIn;
        if (isLoad)
            oldIn = cast<::tla::LoadUnalignUpdateOp>(cloned).getUregIn();
        else
            oldIn = cast<::tla::StoreUnalignUpdateOp>(cloned).getUregIn();
        assignUregIn(cloned, isLoad, prevUreg);
        eraseUnusedUregInit(oldIn);
        prevUreg = uregOutOf(cloned);
    }

    auto oldYield = cast<scf::YieldOp>(oldBody.getTerminator());
    SmallVector<Value> newYieldOperands;
    newYieldOperands.reserve(oldYield.getNumOperands() + 1);
    for (Value v : oldYield.getOperands())
        newYieldOperands.push_back(mapping.lookupOrDefault(v));
    newYieldOperands.push_back(uregOutOf(clonedChain.back()));
    bodyBuilder.create<scf::YieldOp>(oldYield.getLoc(), newYieldOperands);

    for (auto [oldRes, newRes] : llvm::zip(forOp.getResults(), newFor.getResults().drop_back()))
        oldRes.replaceAllUsesWith(newRes);

    // Do not map for.result(tensor) → for.result(ureg): that would enable
    // epilogue / next-for 续流. Post tips via the yielded ureg SSA use instead.
    forOp.erase();
    return success();
}

/// Force dual-instr on a deferred head when elim is not possible.
static void materializeDualInstrHead(Operation* updateOp, bool isLoad, UregMaps& uregMaps)
{
    Value window = windowTensorOf(updateOp, isLoad);
    OpBuilder b(updateOp);
    Type uregTy = ::tla::UregType::get(updateOp->getContext());
    Value uregIn = isLoad ? b.create<::tla::LoadUnalignPreOp>(updateOp->getLoc(), uregTy, window).getResult() :
                            b.create<::tla::CreateUregOp>(updateOp->getLoc(), uregTy).getResult();
    assignUregIn(updateOp, isLoad, uregIn);
    if (Value next = nextWindowTensorFromUpdate(updateOp))
        uregMaps.forDir(isLoad)[next] = uregOutOf(updateOp);
}

/// Rewrite at most one for-carried UnAlign ureg stream. Returns true if rewritten.
static FailureOr<bool> rewriteOneForUregCarry(scf::ForOp forOp, UregMaps& uregMaps)
{
    struct Candidate {
        Operation* update = nullptr;
        unsigned tensorIterIdx = 0;
        bool isLoad = true;
    };
    SmallVector<Candidate, 4> candidates;

    for (Operation& op : *forOp.getBody()) {
        if (auto load = dyn_cast<::tla::LoadUnalignUpdateOp>(&op)) {
            if (load.getUregIn())
                continue;
            if (auto idx = matchIterArgIndex(load.getSource(), forOp)) {
                // WAR store in body: dual-instr (did not defer); never for-carry.
                if (forBodyBlocksLoadUregElim(forOp, &op))
                    continue;
                candidates.push_back({&op, *idx, true});
            }
        } else if (auto store = dyn_cast<::tla::StoreUnalignUpdateOp>(&op)) {
            if (store.getUregIn())
                continue;
            if (auto idx = matchIterArgIndex(store.getDest(), forOp)) {
                // RAW observer in body: dual-instr (did not defer); never for-carry.
                if (forBodyBlocksStoreUregElim(forOp, &op))
                    continue;
                candidates.push_back({&op, *idx, false});
            }
        }
    }
    if (candidates.empty())
        return false;

    const Candidate& c = candidates.front();
    if (llvm::count_if(candidates, [&](const Candidate& o) {
            return o.tensorIterIdx == c.tensorIterIdx && o.isLoad == c.isLoad;
        }) != 1) {
        return c.update->emitError(
                   Twine(kPassDiagPrefix) +
                   "multiple UnAlign UPDATE ops on the same scf.for iter_arg "
                   "without a unique back-edge ureg SSA"),
               failure();
    }

    FailureOr<SmallVector<Operation*, 4>> chainOr = buildUregCarryChain(forOp, c.tensorIterIdx, c.isLoad, c.update);
    if (failed(chainOr))
        return failure();
    SmallVector<Operation*, 4> chain = *chainOr;
    if (chain.empty()) {
        // Yielded tensor is not on a unique chain from this head → dual-instr.
        // Still a change: remaining deferred candidates in this for need another
        // rewrite pass (returning false would leave them without ureg_in).
        materializeDualInstrHead(c.update, c.isLoad, uregMaps);
        return true;
    }

    Value initTensor = forOp.getInitArgs()[c.tensorIterIdx];
    // No cross-for 续流: never wire a prologue (or prior for) ureg as this for's
    // ureg init — always open a fresh Pre/create_ureg at the for.
    OpBuilder b(forOp);
    Type uregTy = ::tla::UregType::get(forOp.getContext());
    Value uregInit = c.isLoad ? b.create<::tla::LoadUnalignPreOp>(forOp.getLoc(), uregTy, initTensor).getResult() :
                                b.create<::tla::CreateUregOp>(forOp.getLoc(), uregTy).getResult();

    if (failed(addUregCarryForChain(forOp, c.tensorIterIdx, chain, uregInit, c.isLoad)))
        return failure();
    return true;
}

static bool uregContinuesIntoStoreUpdate(Value ureg)
{
    // For-carried ureg results are flushed after the for (no cross-for 续流).
    if (isScfForResultUreg(ureg))
        return false;
    for (OpOperand& use : ureg.getUses()) {
        if (auto store = dyn_cast<::tla::StoreUnalignUpdateOp>(use.getOwner())) {
            if (store.getUregIn() != ureg)
                continue;
            // Continuation across a same-root conflict is not a real stream tip:
            // Post must fire before that access (bindOrInsert should have broken
            // the chain; this guards tip placement if a use remains).
            if (!storeUregReuseCrossesConflict(ureg, store.getOperation()))
                return true;
        }
    }
    return false;
}

enum class StoreUregTipKind
{
    Skip,
    Tip,
    Error
};

static StoreUregTipKind resolveStoreUregTip(
    ::tla::StoreUnalignUpdateOp store, Value& tipUreg, Value& tipWindowTensor, Operation*& insertAfter)
{
    Value uregOut = store.getUregOut();
    if (uregContinuesIntoStoreUpdate(uregOut))
        return StoreUregTipKind::Skip;

    Value nextWindow = nextWindowTensorFromUpdate(store.getOperation());
    if (!nextWindow) {
        (void)store.emitError(
            Twine(kPassDiagPrefix) +
            "store UPDATE ureg last-use has no next !tla.tensor window "
            "(make_tensor / make_tensor_like of next_ptr)");
        return StoreUregTipKind::Error;
    }

    SmallVector<OpOperand*, 2> yieldUses;
    for (OpOperand& use : uregOut.getUses()) {
        if (isa<scf::YieldOp>(use.getOwner()))
            yieldUses.push_back(&use);
    }
    if (yieldUses.size() > 1) {
        (void)store.emitError(
            Twine(kPassDiagPrefix) +
            "store UPDATE ureg result has multiple scf.yield uses "
            "(unsupported if/else-style merge of ureg)");
        return StoreUregTipKind::Error;
    }
    if (yieldUses.size() == 1) {
        auto yield = cast<scf::YieldOp>(yieldUses.front()->getOwner());
        auto forOp = dyn_cast<scf::ForOp>(yield->getParentOp());
        if (!forOp) {
            (void)store.emitError(Twine(kPassDiagPrefix) + "store ureg_out yielded outside scf.for (unsupported)");
            return StoreUregTipKind::Error;
        }
        tipUreg = forOp.getResult(yieldUses.front()->getOperandNumber());

        std::optional<unsigned> windowResIdx;
        for (auto [i, y] : llvm::enumerate(yield.getOperands())) {
            if (y == nextWindow)
                windowResIdx = i;
        }
        if (!windowResIdx) {
            (void)store.emitError(
                Twine(kPassDiagPrefix) +
                "store next window tensor is not yielded with ureg_out "
                "(cannot place store_unalign_post)");
            return StoreUregTipKind::Error;
        }
        tipWindowTensor = forOp.getResult(*windowResIdx);
        if (uregContinuesIntoStoreUpdate(tipUreg))
            return StoreUregTipKind::Skip;
        insertAfter = forOp.getOperation();
        return StoreUregTipKind::Tip;
    }

    // Cross-for 续流 (ureg_out as a later for's ureg init) is unsupported:
    // prologue always opens a fresh create_ureg, so this path is unreachable
    // from frontend lowering and must fail closed on hand-written IR.
    for (OpOperand& use : uregOut.getUses()) {
        if (isa<::tla::StoreUnalignPostOp>(use.getOwner()))
            continue;
        if (auto forOp = dyn_cast<scf::ForOp>(use.getOwner())) {
            unsigned opIdx = use.getOperandNumber();
            if (opIdx >= forOp.getNumControlOperands()) {
                (void)store.emitError(
                    Twine(kPassDiagPrefix) +
                    "store ureg_out used as scf.for ureg init is unsupported "
                    "(no cross-for 续流; expect create_ureg at the for)");
                return StoreUregTipKind::Error;
            }
        }
        (void)store.emitError(
            Twine(kPassDiagPrefix) + "store ureg_out has a non-yield user at ureg last-use (unsupported)");
        return StoreUregTipKind::Error;
    }

    tipUreg = uregOut;
    tipWindowTensor = nextWindow;
    insertAfter = nextWindow.getDefiningOp();
    return StoreUregTipKind::Tip;
}

static LogicalResult insertStoreUregPosts(func::FuncOp funcOp)
{
    SmallVector<::tla::StoreUnalignUpdateOp, 8> stores;
    funcOp.walk([&](::tla::StoreUnalignUpdateOp op) { stores.push_back(op); });

    DenseSet<Value> postedTips;
    // Anchor op → last Post inserted for tips that share that anchor. Repeated
    // ``setInsertionPointAfter(for)`` would reverse order; append after the
    // previous Post instead (尾插, program/walk order).
    DenseMap<Operation*, Operation*> lastPostAfterAnchor;
    for (::tla::StoreUnalignUpdateOp store : stores) {
        Value tipUreg, tipWindowTensor;
        Operation* insertAfter = nullptr;
        switch (resolveStoreUregTip(store, tipUreg, tipWindowTensor, insertAfter)) {
            case StoreUregTipKind::Skip:
                continue;
            case StoreUregTipKind::Error:
                return failure();
            case StoreUregTipKind::Tip:
                break;
        }
        if (!postedTips.insert(tipUreg).second)
            continue;
        if (llvm::any_of(tipUreg.getUsers(), [](Operation* u) { return isa<::tla::StoreUnalignPostOp>(u); }))
            continue;
        Operation* anchor = insertAfter;
        if (Operation* prev = lastPostAfterAnchor.lookup(anchor))
            insertAfter = prev;
        OpBuilder b(insertAfter);
        b.setInsertionPointAfter(insertAfter);
        auto post = b.create<::tla::StoreUnalignPostOp>(store.getLoc(), tipUreg, tipWindowTensor);
        lastPostAfterAnchor[anchor] = post.getOperation();
    }
    return success();
}

static LogicalResult runOnFunction(func::FuncOp funcOp)
{
    UregMaps uregMaps;
    bool changed = true;
    while (changed) {
        changed = false;
        // Drop any window→ureg entries from a prior iteration. A successful
        // for-carry rewrite erases the old for body; keeping those Values in
        // the maps is use-after-free if Operation storage is recycled.
        uregMaps.clear();
        funcOp.walk([&](Block* block) { fillUregInBlock(*block, uregMaps); });

        SmallVector<scf::ForOp, 4> fors;
        funcOp.walk([&](scf::ForOp forOp) { fors.push_back(forOp); });
        llvm::sort(
            fors, [](scf::ForOp a, scf::ForOp b) { return b.getOperation()->isProperAncestor(a.getOperation()); });

        for (scf::ForOp forOp : fors) {
            if (!forOp->getBlock())
                continue;
            FailureOr<bool> did = rewriteOneForUregCarry(forOp, uregMaps);
            if (failed(did))
                return failure();
            if (*did) {
                changed = true;
                break;
            }
        }
    }

    uregMaps.clear();
    funcOp.walk([&](Block* block) { fillUregInBlock(*block, uregMaps); });

    LogicalResult status = success();
    funcOp.walk([&](Operation* op) {
        auto missing = [&](Operation* o) {
            (void)o->emitError(Twine(kPassDiagPrefix) + "UPDATE missing ureg_in after rewrite");
            status = failure();
        };
        if (auto load = dyn_cast<::tla::LoadUnalignUpdateOp>(op)) {
            if (!load.getUregIn())
                missing(op);
        } else if (auto store = dyn_cast<::tla::StoreUnalignUpdateOp>(op)) {
            if (!store.getUregIn())
                missing(op);
        }
    });
    if (failed(status))
        return failure();
    return insertStoreUregPosts(funcOp);
}

class TlaUnalignUregPass : public PassWrapper<TlaUnalignUregPass, OperationPass<ModuleOp>> {
public:
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TlaUnalignUregPass)

    StringRef getArgument() const final
    {
        return "tla-unalign-ureg";
    }
    StringRef getDescription() const final
    {
        return "Fill UnAlign UPDATE ureg input; insert load_unalign_pre / "
               "create_ureg / store_unalign_post";
    }

    void runOnOperation() override
    {
        for (func::FuncOp funcOp : getOperation().getOps<func::FuncOp>()) {
            if (failed(runOnFunction(funcOp))) {
                signalPassFailure();
                return;
            }
        }
    }
};

} // namespace

std::unique_ptr<Pass> createTlaUnalignUregPass()
{
    return std::make_unique<TlaUnalignUregPass>();
}

void registerTlaUnalignUregPass()
{
    PassRegistration<TlaUnalignUregPass>();
}

} // namespace tla
