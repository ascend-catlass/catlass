#include "PassesCommon.h"
#include "PassesInternal.h"

namespace tla {
namespace {

// Implicit, coalesced `hivm.set_ctrl` for every DSL op whose semantics depend on
// a CTRL bit. This pass is the single place that encodes those dependencies:
// InitSocState alone is not "done" — even when InitSoc already leaves the bit
// in the required value, the requirement is still registered here and
// `ensureCtrlBit` simply no-ops (no insert).
//
// Inventory (InitSoc bulk CTRL defaults after entry):
//   CTRL[60]=1  OverrideSaturation (global; instruction sat ignored)
//   CTRL[51]=0  mmad compute-order (M_FIRST)
//   CTRL[46]=0  mmad HF32 enable
//   CTRL[47]=0  mmad HF32 rounding select
//   CTRL[48]    preserved (not owned here; VCVT uses instr sat when 60=0)
//   CTRL[56]    replaced by InitSoc mask_norm / vector_mask (no DSL API)
//
// Clients:
//   tla.cast  sat_mode ∈ {SAT, NOSAT, UNKNOWN} → CTRL[60]=0
//             (instruction sat operand must be honored; UNKNOWN lowers as NOSAT)
//   tla.mmad / tla.mmad_mx compute_order → CTRL[51]
//   tla.mmad hf32_mode → CTRL[46] + CTRL[47]
//
// Control-flow:
//   Uniform reqs on a container → hoist SetCtrl before it; exit state stays known.
//   scf.if with Mixed → analyze each region from a copy of the entry state; join
//     disagreeing bits as unknown.
//   scf.for/while with Mixed → mark only Mixed bits unknown at the header (back
//     edge), analyze the body, then mark those bits unknown again at the exit.
//   Unknown bits force the next ensure to re-set.

static constexpr unsigned OverrideSaturationBit = 60;
static constexpr unsigned ComputeOrderBit = 51;
static constexpr unsigned HF32ModeBit = 46;
static constexpr unsigned HF32TransModeBit = 47;

static constexpr unsigned kTrackedCtrlBits[] = {OverrideSaturationBit, ComputeOrderBit, HF32ModeBit, HF32TransModeBit};

static constexpr unsigned kSatModeTraitIndex = 1;

enum class CtrlBitReq
{
    None,  // no dependency
    Zero,  // needs the bit cleared
    One,   // needs the bit set
    Mixed, // nested ops need both; insert transitions inside
};

static CtrlBitReq mergeReq(CtrlBitReq a, CtrlBitReq b)
{
    if (a == CtrlBitReq::None)
        return b;
    if (b == CtrlBitReq::None)
        return a;
    if (a == b)
        return a;
    return CtrlBitReq::Mixed;
}

using BitReqs = llvm::SmallDenseMap<unsigned, CtrlBitReq, 4>;
using CtrlState = llvm::SmallDenseMap<unsigned, std::optional<bool>, 4>;

static void mergeBitReqs(BitReqs& dst, const BitReqs& src)
{
    for (auto& kv : src) {
        CtrlBitReq prev = CtrlBitReq::None;
        auto it = dst.find(kv.first);
        if (it != dst.end())
            prev = it->second;
        dst[kv.first] = mergeReq(prev, kv.second);
    }
}

static void setBitReq(BitReqs& reqs, unsigned bit, bool enable)
{
    CtrlBitReq add = enable ? CtrlBitReq::One : CtrlBitReq::Zero;
    CtrlBitReq prev = CtrlBitReq::None;
    auto it = reqs.find(bit);
    if (it != reqs.end())
        prev = it->second;
    reqs[bit] = mergeReq(prev, add);
}

static bool isStructuredCtrlContainer(Operation* op)
{
    return isa<::tla::VectorOp, ::tla::VecFuncOp, ::tla::CubeOp, scf::ForOp, scf::WhileOp, scf::IfOp>(op);
}

static bool isTrackedCtrlBit(unsigned bit)
{
    for (unsigned tracked : kTrackedCtrlBits) {
        if (tracked == bit)
            return true;
    }
    return false;
}

static std::optional<bool> getBitState(const CtrlState& state, unsigned bit)
{
    auto it = state.find(bit);
    if (it == state.end())
        return std::nullopt;
    return it->second;
}

// Every cast that carries a sat_mode trait relies on the instruction sat
// operand, so OverrideSaturation must be off. InitSoc leaves CTRL[60]=1.
static void castCtrlReqs(::tla::CastOp castOp, BitReqs& reqs)
{
    ArrayRef<int32_t> trait = castOp.getTrait();
    if (trait.size() <= kSatModeTraitIndex)
        return;
    // SAT / NOSAT / UNKNOWN all encode an instruction sat operand; UNKNOWN
    // lowers as NOSAT. OverrideSaturation must be off for that operand to apply.
    setBitReq(reqs, OverrideSaturationBit, /*enable=*/false);
}

static void mmadCtrlReqs(Operation* op, BitReqs& reqs)
{
    auto orderAttr = op->getAttrOfType<::tla::ComputeOrderAttr>("compute_order");
    if (orderAttr) {
        bool nFirst = orderAttr.getValue() == ComputeOrder::N_FIRST;
        setBitReq(reqs, ComputeOrderBit, /*enable=*/nFirst);
    }
    // hf32_mode exists only on tla.mmad (not mmad_mx).
    if (auto modeAttr = op->getAttrOfType<::tla::HF32ModeAttr>("hf32_mode")) {
        HF32Mode mode = modeAttr.getValue();
        bool enableHF32 = mode != HF32Mode::HF32_DISABLE;
        bool nearestZero = mode == HF32Mode::HF32_NEAREST_ZERO;
        setBitReq(reqs, HF32ModeBit, /*enable=*/enableHF32);
        setBitReq(reqs, HF32TransModeBit, /*enable=*/nearestZero);
    }
}

static BitReqs reqOfOp(Operation* op)
{
    BitReqs reqs;
    if (auto castOp = dyn_cast<::tla::CastOp>(op))
        castCtrlReqs(castOp, reqs);
    else if (isa<::tla::MmadOp, ::tla::MmadMxOp>(op))
        mmadCtrlReqs(op, reqs);
    return reqs;
}

static BitReqs reqOfRegion(Region& region)
{
    BitReqs reqs;
    region.walk([&](Operation* op) {
        if (isStructuredCtrlContainer(op))
            return;
        mergeBitReqs(reqs, reqOfOp(op));
    });
    return reqs;
}

static void ensureCtrlBit(Operation* before, unsigned bit, bool enable, CtrlState& state, OpBuilder& builder)
{
    auto it = state.find(bit);
    if (it != state.end() && it->second.has_value() && *it->second == enable)
        return;
    builder.setInsertionPoint(before);
    builder.create<hivm::SetCtrlOp>(before->getLoc(), enable, bit);
    state[bit] = enable;
}

static void ensureReqs(Operation* before, const BitReqs& reqs, CtrlState& state, OpBuilder& builder)
{
    // Stable ascending bit order so lit / IR dumps are deterministic
    // (SmallDenseMap iteration order is not).
    SmallVector<std::pair<unsigned, CtrlBitReq>, 4> ordered(reqs.begin(), reqs.end());
    llvm::sort(ordered, [](const auto& a, const auto& b) { return a.first < b.first; });
    for (auto& kv : ordered) {
        if (kv.second == CtrlBitReq::Zero)
            ensureCtrlBit(before, kv.first, /*enable=*/false, state, builder);
        else if (kv.second == CtrlBitReq::One)
            ensureCtrlBit(before, kv.first, /*enable=*/true, state, builder);
    }
}

static bool hasMixedReq(const BitReqs& reqs)
{
    for (auto& kv : reqs) {
        if (kv.second == CtrlBitReq::Mixed)
            return true;
    }
    return false;
}

static void invalidateMixedBits(CtrlState& state, const BitReqs& reqs)
{
    for (auto& kv : reqs) {
        if (kv.second == CtrlBitReq::Mixed)
            state[kv.first] = std::nullopt;
    }
}

static BitReqs uniformReqsOf(const BitReqs& reqs)
{
    BitReqs uniform;
    for (auto& kv : reqs) {
        if (kv.second == CtrlBitReq::Zero || kv.second == CtrlBitReq::One)
            uniform[kv.first] = kv.second;
    }
    return uniform;
}

// Join scf.if then/else exits: agreeing concrete values are kept; anything that
// disagrees or is already unknown becomes unknown. scf.if contributes then +
// else (empty else still counts); fewer exits → treat all tracked bits unknown.
static void mergeBranchStates(CtrlState& state, ArrayRef<CtrlState> exits)
{
    if (exits.size() < 2) {
        for (unsigned bit : kTrackedCtrlBits)
            state[bit] = std::nullopt;
        return;
    }
    for (unsigned bit : kTrackedCtrlBits) {
        std::optional<bool> agreed = getBitState(exits[0], bit);
        if (agreed != getBitState(exits[1], bit))
            agreed = std::nullopt;
        state[bit] = agreed;
    }
}

static void processBlock(Block& block, CtrlState& state, OpBuilder& builder);

static void processRegions(Operation* op, CtrlState& state, OpBuilder& builder)
{
    for (Region& region : op->getRegions()) {
        for (Block& nested : region)
            processBlock(nested, state, builder);
    }
}

static void processStructuredOp(Operation* op, CtrlState& state, OpBuilder& builder)
{
    BitReqs reqs;
    for (Region& region : op->getRegions())
        mergeBitReqs(reqs, reqOfRegion(region));

    // Mutually exclusive regions: never thread one branch's exit into another.
    if (isa<scf::IfOp>(op) && hasMixedReq(reqs)) {
        CtrlState entryState = state;
        SmallVector<CtrlState, 2> exits;
        exits.reserve(op->getNumRegions());
        for (Region& region : op->getRegions()) {
            CtrlState branchState = entryState;
            for (Block& nested : region)
                processBlock(nested, branchState, builder);
            exits.push_back(std::move(branchState));
        }
        mergeBranchStates(state, exits);
        return;
    }

    // Loops with toggling CTRL needs:
    // - Hoist Uniform (non-Mixed) bits before the loop so a zero-trip path still
    //   establishes them and post-loop code can rely on a known value.
    // - Mark Mixed bits unknown at the header (back edge) and again at the exit.
    // Fully-uniform loops do not take this path (they hoist all bits below).
    if (isa<scf::ForOp, scf::WhileOp>(op) && hasMixedReq(reqs)) {
        BitReqs uniformReqs = uniformReqsOf(reqs);
        if (!uniformReqs.empty())
            ensureReqs(op, uniformReqs, state, builder);
        invalidateMixedBits(state, reqs);
        processRegions(op, state, builder);
        invalidateMixedBits(state, reqs);
        return;
    }

    if (hasMixedReq(reqs)) {
        // Single-region containers (vector / cube / vec.func): straight-line
        // transitions inside; sequential state threading is correct.
        processRegions(op, state, builder);
        return;
    }

    if (!reqs.empty()) {
        // Uniform requirements for the whole container: one SetCtrl per bit
        // before the container. Exit state remains the ensured values (including
        // after for/while — the write runs outside the loop regardless of trip
        // count).
        ensureReqs(op, reqs, state, builder);
        return;
    }

    processRegions(op, state, builder);
}

static void processBlock(Block& block, CtrlState& state, OpBuilder& builder)
{
    for (Operation& opRef : llvm::make_early_inc_range(block)) {
        Operation* op = &opRef;

        if (auto setCtrl = dyn_cast<hivm::SetCtrlOp>(op)) {
            if (isTrackedCtrlBit(setCtrl.getIdx()))
                state[setCtrl.getIdx()] = setCtrl.getEnable();
            continue;
        }

        if (isStructuredCtrlContainer(op)) {
            processStructuredOp(op, state, builder);
            continue;
        }

        ensureReqs(op, reqOfOp(op), state, builder);
    }
}

static CtrlState initSocCtrlState()
{
    // Matches InitSoc bulk CTRL reset (injected later by prologue).
    CtrlState state;
    state[OverrideSaturationBit] = true;
    state[ComputeOrderBit] = false;
    state[HF32ModeBit] = false;
    state[HF32TransModeBit] = false;
    return state;
}

class TlaInsertImplicitCtrlPass : public PassWrapper<TlaInsertImplicitCtrlPass, OperationPass<ModuleOp>> {
public:
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TlaInsertImplicitCtrlPass)

    StringRef getArgument() const override
    {
        return "tla-insert-implicit-ctrl";
    }
    StringRef getName() const override
    {
        return "TlaInsertImplicitCtrlPass";
    }
    StringRef getDescription() const override
    {
        return "Insert coalesced hivm.set_ctrl for ops that implicitly depend on CTRL bits.";
    }

    void getDependentDialects(DialectRegistry& registry) const override
    {
        registry.insert<hivm::HIVMDialect, ::tla::TlaDialect, scf::SCFDialect, func::FuncDialect>();
    }

    void runOnOperation() override
    {
        ModuleOp module = getOperation();
        OpBuilder builder(&getContext());

        for (func::FuncOp funcOp : module.getOps<func::FuncOp>()) {
            if (funcOp.isDeclaration())
                continue;
            // Outlined vector helpers are created later; place SetCtrl in the
            // parent so a loop of CTRL-dependent ops does not re-set every iter.
            if (funcOp->hasAttr(kHivmVectorFunctionAttrName))
                continue;

            CtrlState state = initSocCtrlState();
            for (Block& block : funcOp.getBody())
                processBlock(block, state, builder);
        }
    }
};

} // namespace

std::unique_ptr<Pass> createTlaInsertImplicitCtrlPass()
{
    return std::make_unique<TlaInsertImplicitCtrlPass>();
}

void registerTlaInsertImplicitCtrlPass()
{
    PassRegistration<TlaInsertImplicitCtrlPass>();
}

} // namespace tla
