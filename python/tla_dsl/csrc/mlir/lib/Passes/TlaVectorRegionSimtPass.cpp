#include "Passes/TlaDebugPrint.h"
#include "PassesCommon.h"
#include "PassesInternal.h"
#include "Passes/TlaTensorToMemref.h"
#include "Passes/TlaVectorRegionCommon.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/HIVMAVE/IR/HIVMAVE.h"
#include "bishengir/Dialect/Utils/Util.h"
#include "llvm/ADT/StringSwitch.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include <algorithm>
#include <limits>

namespace tla {
using namespace ::tla::vecregion;
namespace {

// Lower a TLA op with three i32 results onto three separate regbase intrinsics,
// one per component. The intrinsics are modelled as independent nullary ops
// (LLVM intrinsics cannot return an aggregate), so a result nobody reads simply
// never gets an intrinsic emitted for it -- `tid, _, _ = tla.arch.thread_idx()`
// costs exactly one instruction.
template <typename TlaOpTy, typename XOpTy, typename YOpTy, typename ZOpTy>
class LowerSimtTripleOpPattern : public OpRewritePattern<TlaOpTy> {
public:
    using OpRewritePattern<TlaOpTy>::OpRewritePattern;

    LogicalResult matchAndRewrite(TlaOpTy op, PatternRewriter& rewriter) const override
    {
        Location loc = op.getLoc();
        Type i32 = rewriter.getI32Type();

        if (op.getX().use_empty() && op.getY().use_empty() && op.getZ().use_empty()) {
            rewriter.eraseOp(op);
            return success();
        }

        // Replace uses individually rather than via replaceOp: that way an unused
        // component needs no placeholder value and no intrinsic.
        if (!op.getX().use_empty())
            rewriter.replaceAllUsesWith(op.getX(), rewriter.create<XOpTy>(loc, i32).getRes());
        if (!op.getY().use_empty())
            rewriter.replaceAllUsesWith(op.getY(), rewriter.create<YOpTy>(loc, i32).getRes());
        if (!op.getZ().use_empty())
            rewriter.replaceAllUsesWith(op.getZ(), rewriter.create<ZOpTy>(loc, i32).getRes());
        rewriter.eraseOp(op);
        return success();
    }
};

using LowerThreadIdxPattern = LowerSimtTripleOpPattern<
    ::tla::ThreadIdxOp, hivm_regbaseintrins::ThreadIdXOp, hivm_regbaseintrins::ThreadIdYOp,
    hivm_regbaseintrins::ThreadIdZOp>;

using LowerBlockDimPattern = LowerSimtTripleOpPattern<
    ::tla::ThreadBlockDimOp, hivm_regbaseintrins::BlockDimXOp, hivm_regbaseintrins::BlockDimYOp,
    hivm_regbaseintrins::BlockDimZOp>;

// ---------------------------------------------------------------------------
// SIMT vec.func outlining
//
// A SIMT region is per-thread scalar code, not whole-vector AVE work, so it is
// outlined into its own function and invoked through
// hivm_regbaseintrins.intrins.launch_func rather than being folded into the
// enclosing AIV body. The outlined function must stay in the plain
// memref/scf/arith form that hivmc-a5 accepts for a simt_entry function -- no
// reinterpret_cast views, no descriptor arithmetic. Everything the body needs is
// therefore materialized *before* the launch and handed over as arguments:
// buffers as raw pointers (extract_aligned_pointer_as_index -> index_cast ->
// llvm.inttoptr), thread geometry as the launch's thread_block_dim triple.
// ---------------------------------------------------------------------------

// Values produced outside `region` but used inside it. Constant-like producers
// are reported separately: they are cloned into the outlined body instead of
// consuming an ABI slot.
struct SimtCaptures {
    SmallVector<Value, 4> tensors;        // tla.tensor values reached by scalar access or printing
    SmallVector<Value, 4> scalars;        // everything else that must be passed in
    SmallVector<Operation*, 8> constants; // clone into the body
};

static bool isDefinedOutside(Value value, Region& region)
{
    Region* defRegion = value.getParentRegion();
    return !defRegion || !region.isAncestor(defRegion);
}

static void collectSimtCaptures(::tla::VecFuncOp vecFuncOp, SimtCaptures& captures)
{
    Region& region = vecFuncOp.getBody();
    SetVector<Value> tensorSet;
    SetVector<Value> scalarSet;
    SetVector<Operation*> constantSet;

    region.walk([&](Operation* op) {
        // Tensor operands of scalar accesses and prints become memref parameters.
        if (auto load = dyn_cast<::tla::SimtLoadOp>(op)) {
            if (isDefinedOutside(load.getSource(), region))
                tensorSet.insert(load.getSource());
        } else if (auto store = dyn_cast<::tla::SimtStoreOp>(op)) {
            if (isDefinedOutside(store.getDest(), region))
                tensorSet.insert(store.getDest());
        } else if (auto print = dyn_cast<::tla::PrintTensorOp>(op)) {
            if (isDefinedOutside(print.getValue(), region))
                tensorSet.insert(print.getValue());
        }
        for (Value operand : op->getOperands()) {
            if (!isDefinedOutside(operand, region))
                continue;
            if (tensorSet.contains(operand) || isa<::tla::TlaTensorType>(operand.getType()))
                continue;
            if (Operation* def = operand.getDefiningOp()) {
                if (def->hasTrait<mlir::OpTrait::ConstantLike>()) {
                    constantSet.insert(def);
                    continue;
                }
            }
            scalarSet.insert(operand);
        }
    });

    captures.tensors.assign(tensorSet.begin(), tensorSet.end());
    captures.scalars.assign(scalarSet.begin(), scalarSet.end());
    captures.constants.assign(constantSet.begin(), constantSet.end());
}

// Same fan-out as LowerSimtTripleOpPattern, applied directly to a finished
// function. The outlining clones the region before the rewrite driver reaches
// its ops, so the identity queries have to be lowered here rather than left to
// the pattern -- the outlined function is not re-driven.
template <typename TlaOpTy, typename XOpTy, typename YOpTy, typename ZOpTy>
static void lowerSimtTripleOpsIn(func::FuncOp vf)
{
    SmallVector<TlaOpTy, 4> ops;
    vf.walk([&](TlaOpTy op) { ops.push_back(op); });
    for (TlaOpTy op : ops) {
        OpBuilder builder(op);
        Location loc = op.getLoc();
        Type i32 = builder.getI32Type();
        if (!op.getX().use_empty())
            op.getX().replaceAllUsesWith(builder.create<XOpTy>(loc, i32).getRes());
        if (!op.getY().use_empty())
            op.getY().replaceAllUsesWith(builder.create<YOpTy>(loc, i32).getRes());
        if (!op.getZ().use_empty())
            op.getZ().replaceAllUsesWith(builder.create<ZOpTy>(loc, i32).getRes());
        op.erase();
    }
}

// Address spaces the SIMT ABI addresses device buffers through, on both the
// launch's !llvm.ptr operands and the outlined function's memref parameters.
// The two must agree or hivmc-a5 rejects the call signature. These are the
// integer forms: #hivm.address_space<...> lowers to a descriptor struct rather
// than a bare pointer, which does not match what launch_func passes.
static constexpr int kSimtGmAddressSpace = 1;
static constexpr int kSimtUbAddressSpace = 6;

// Integer address space for a descriptor's address space name, or failure for
// one a SIMT vector function cannot reach.
static std::optional<int> simtAddressSpaceFor(StringRef addrspace)
{
    if (addrspace == "gm")
        return kSimtGmAddressSpace;
    if (addrspace == "ub")
        return kSimtUbAddressSpace;
    return std::nullopt;
}

// The per-thread scalar arithmetic ops carry their own identity through the TLA
// pipeline so a SIMT body reads as TLA IR rather than raw arith; here each one
// becomes the arith op for its element type. Integer division uses the signed
// form: the frontend only emits tla.simt_div for floats (Numeric '/' rejects
// integers), so an integer divide can only arrive from hand-written TLA IR.
// tla.simt_cast -> the arith conversion its `kind` names.
static void lowerSimtCastIn(func::FuncOp vf)
{
    SmallVector<::tla::SimtCastOp, 8> ops;
    vf.walk([&](::tla::SimtCastOp op) { ops.push_back(op); });
    for (::tla::SimtCastOp op : ops) {
        OpBuilder builder(op);
        Location loc = op.getLoc();
        Type resultType = op.getResult().getType();
        Value src = op.getSource();
        StringRef kind = op.getKind();
        Value lowered;
        if (kind == "extsi")
            lowered = builder.create<arith::ExtSIOp>(loc, resultType, src);
        else if (kind == "extui")
            lowered = builder.create<arith::ExtUIOp>(loc, resultType, src);
        else if (kind == "trunci")
            lowered = builder.create<arith::TruncIOp>(loc, resultType, src);
        else if (kind == "extf")
            // llvm.fpext / llvm.fptrunc rather than the arith ops: for bf16
            // convert-hivm-to-std rewrites those into vcast_*_1d_with_mode
            // vector-template calls that the c310 bitcode does not export.
            lowered = builder.create<LLVM::FPExtOp>(loc, resultType, src);
        else if (kind == "truncf")
            lowered = builder.create<LLVM::FPTruncOp>(loc, resultType, src);
        else if (kind == "sitofp")
            lowered = builder.create<arith::SIToFPOp>(loc, resultType, src);
        else if (kind == "uitofp")
            lowered = builder.create<arith::UIToFPOp>(loc, resultType, src);
        else if (kind == "fptosi")
            lowered = builder.create<arith::FPToSIOp>(loc, resultType, src);
        else
            lowered = builder.create<arith::FPToUIOp>(loc, resultType, src);
        op.getResult().replaceAllUsesWith(lowered);
        op.erase();
    }
}

// tla.simt_where -> arith.select. The condition is already an i1, so the
// operands carry the result type and the plain builder suffices.
static void lowerSimtWhereIn(func::FuncOp vf)
{
    SmallVector<::tla::SimtWhereOp, 8> ops;
    vf.walk([&](::tla::SimtWhereOp op) { ops.push_back(op); });
    for (::tla::SimtWhereOp op : ops) {
        OpBuilder builder(op);
        Value lowered =
            builder.create<arith::SelectOp>(op.getLoc(), op.getCondition(), op.getX(), op.getY()).getResult();
        op.getResult().replaceAllUsesWith(lowered);
        op.erase();
    }
}

// tla.simt_cmp -> arith.cmpf (ordered) / arith.cmpi, picking the signed or
// unsigned integer predicate the frontend recorded.
static void lowerSimtCmpIn(func::FuncOp vf)
{
    SmallVector<::tla::SimtCmpOp, 8> ops;
    vf.walk([&](::tla::SimtCmpOp op) { ops.push_back(op); });
    for (::tla::SimtCmpOp op : ops) {
        OpBuilder builder(op);
        Location loc = op.getLoc();
        StringRef mode = op.getMode();
        Value lowered;
        if (isa<FloatType>(op.getLhs().getType())) {
            arith::CmpFPredicate pred = llvm::StringSwitch<arith::CmpFPredicate>(mode)
                                            .Case("lt", arith::CmpFPredicate::OLT)
                                            .Case("le", arith::CmpFPredicate::OLE)
                                            .Case("gt", arith::CmpFPredicate::OGT)
                                            .Case("ge", arith::CmpFPredicate::OGE)
                                            .Case("eq", arith::CmpFPredicate::OEQ)
                                            .Default(arith::CmpFPredicate::ONE);
            lowered = builder.create<arith::CmpFOp>(loc, pred, op.getLhs(), op.getRhs()).getResult();
        } else {
            bool isUnsigned = op.getIsUnsigned();
            arith::CmpIPredicate pred =
                llvm::StringSwitch<arith::CmpIPredicate>(mode)
                    .Case("lt", isUnsigned ? arith::CmpIPredicate::ult : arith::CmpIPredicate::slt)
                    .Case("le", isUnsigned ? arith::CmpIPredicate::ule : arith::CmpIPredicate::sle)
                    .Case("gt", isUnsigned ? arith::CmpIPredicate::ugt : arith::CmpIPredicate::sgt)
                    .Case("ge", isUnsigned ? arith::CmpIPredicate::uge : arith::CmpIPredicate::sge)
                    .Case("eq", arith::CmpIPredicate::eq)
                    .Default(arith::CmpIPredicate::ne);
            lowered = builder.create<arith::CmpIOp>(loc, pred, op.getLhs(), op.getRhs()).getResult();
        }
        op.getResult().replaceAllUsesWith(lowered);
        op.erase();
    }
}

// Same as lowerSimtBinaryIn, but for LLVM intrinsic ops, whose builders need
// the result type spelled out.
template <typename TlaOpTy, typename FloatOpTy, typename IntOpTy>
static void lowerSimtIntrinBinaryIn(func::FuncOp vf)
{
    SmallVector<TlaOpTy, 8> ops;
    vf.walk([&](TlaOpTy op) { ops.push_back(op); });
    for (TlaOpTy op : ops) {
        OpBuilder builder(op);
        Location loc = op.getLoc();
        Type resultType = op.getResult().getType();
        Value lowered = isa<FloatType>(resultType) ?
                            builder.create<FloatOpTy>(loc, resultType, op.getLhs(), op.getRhs()).getResult() :
                            builder.create<IntOpTy>(loc, resultType, op.getLhs(), op.getRhs()).getResult();
        op.getResult().replaceAllUsesWith(lowered);
        op.erase();
    }
}

template <typename TlaOpTy, typename FloatOpTy, typename IntOpTy>
static void lowerSimtBinaryIn(func::FuncOp vf)
{
    SmallVector<TlaOpTy, 8> ops;
    vf.walk([&](TlaOpTy op) { ops.push_back(op); });
    for (TlaOpTy op : ops) {
        OpBuilder builder(op);
        Location loc = op.getLoc();
        Value lowered = isa<FloatType>(op.getResult().getType()) ?
                            builder.create<FloatOpTy>(loc, op.getLhs(), op.getRhs()).getResult() :
                            builder.create<IntOpTy>(loc, op.getLhs(), op.getRhs()).getResult();
        op.getResult().replaceAllUsesWith(lowered);
        op.erase();
    }
}

// The math ops are float-only, so they need no int/float dispatch -- one op in,
// one op out.
// The target has no bf16 transcendental unit -- exp, log, sqrt and pow
// all fail to select -- so those are computed in f32 and rounded back. For
// every other type the compute type is the source type and both casts below
// are omitted, so this costs nothing. The f32 intermediate is strictly more
// accurate than a native bf16 evaluation would be.
static Type simtMathComputeType(Type type)
{
    if (isa<BFloat16Type>(type))
        return Float32Type::get(type.getContext());
    return type;
}

static Value promoteForSimtMath(OpBuilder& builder, Location loc, Value value)
{
    Type computeType = simtMathComputeType(value.getType());
    if (computeType == value.getType())
        return value;
    // llvm.fpext, not arith.extf: convert-hivm-to-std rewrites the arith form
    // into a vcast_bfloat16_t_to_float_1d_with_mode vector-template call that the
    // c310 bitcode does not export.
    return builder.create<LLVM::FPExtOp>(loc, computeType, value);
}

static Value demoteAfterSimtMath(OpBuilder& builder, Location loc, Value value, Type resultType)
{
    if (value.getType() == resultType)
        return value;
    return builder.create<LLVM::FPTruncOp>(loc, resultType, value);
}

template <typename TlaOpTy, typename MathOpTy>
static void lowerSimtUnaryIn(func::FuncOp vf, bool promoteBF16 = true)
{
    SmallVector<TlaOpTy, 8> ops;
    vf.walk([&](TlaOpTy op) { ops.push_back(op); });
    for (TlaOpTy op : ops) {
        OpBuilder builder(op);
        Location loc = op.getLoc();
        Type resultType = op.getResult().getType();
        Value operand = promoteBF16 ? promoteForSimtMath(builder, loc, op.getOperand()) : op.getOperand();
        Value computed = builder.create<MathOpTy>(loc, operand).getResult();
        Value lowered = promoteBF16 ? demoteAfterSimtMath(builder, loc, computed, resultType) : computed;
        op.getResult().replaceAllUsesWith(lowered);
        op.erase();
    }
}

// Lower every tla.simt_* arithmetic op in the outlined vector function.
static void lowerSimtArithmeticIn(func::FuncOp vf)
{
    lowerSimtBinaryIn<::tla::SimtAddOp, arith::AddFOp, arith::AddIOp>(vf);
    lowerSimtBinaryIn<::tla::SimtSubOp, arith::SubFOp, arith::SubIOp>(vf);
    lowerSimtBinaryIn<::tla::SimtMulOp, arith::MulFOp, arith::MulIOp>(vf);
    lowerSimtBinaryIn<::tla::SimtDivOp, arith::DivFOp, arith::DivSIOp>(vf);
    // fmin/fmax semantics: the NaN-propagating forms have no instruction here.
    lowerSimtIntrinBinaryIn<::tla::SimtMaxOp, LLVM::MaxNumOp, LLVM::SMaxOp>(vf);
    lowerSimtIntrinBinaryIn<::tla::SimtMinOp, LLVM::MinNumOp, LLVM::SMinOp>(vf);
    lowerSimtCmpIn(vf);
    lowerSimtWhereIn(vf);
    lowerSimtCastIn(vf);

    SmallVector<::tla::SimtPowOp, 4> powOps;
    vf.walk([&](::tla::SimtPowOp op) { powOps.push_back(op); });
    for (::tla::SimtPowOp op : powOps) {
        OpBuilder builder(op);
        Location powLoc = op.getLoc();
        Type powResultType = op.getResult().getType();
        Value powLhs = promoteForSimtMath(builder, powLoc, op.getLhs());
        Value powRhs = promoteForSimtMath(builder, powLoc, op.getRhs());
        Value lowered = demoteAfterSimtMath(
            builder, powLoc, builder.create<math::PowFOp>(powLoc, powLhs, powRhs).getResult(), powResultType);
        op.getResult().replaceAllUsesWith(lowered);
        op.erase();
    }

    lowerSimtUnaryIn<::tla::SimtSqrtOp, math::SqrtOp>(vf);
    lowerSimtUnaryIn<::tla::SimtExpOp, math::ExpOp>(vf);
    lowerSimtUnaryIn<::tla::SimtAbsOp, math::AbsFOp>(vf, /*promoteBF16=*/false);

    // log goes straight to the LLVM intrinsic: math.log would be expanded by
    // convert-hivm-to-std into a vln_1d_float call that the c310 bitcode does
    // not export.
    SmallVector<::tla::SimtLogOp, 4> logOps;
    vf.walk([&](::tla::SimtLogOp op) { logOps.push_back(op); });
    for (::tla::SimtLogOp op : logOps) {
        OpBuilder builder(op);
        Location logLoc = op.getLoc();
        Type logResultType = op.getResult().getType();
        Value logOperand = promoteForSimtMath(builder, logLoc, op.getOperand());
        Value lowered = demoteAfterSimtMath(
            builder, logLoc, builder.create<LLVM::LogOp>(logLoc, logOperand.getType(), logOperand).getResult(),
            logResultType);
        op.getResult().replaceAllUsesWith(lowered);
        op.erase();
    }
}

// Raw device pointer for a memref, the three steps the SIMT ABI expects.
static Value memrefToRawPointer(OpBuilder& builder, Location loc, Value memref, int addressSpace)
{
    Value index = builder.create<mlir::memref::ExtractAlignedPointerAsIndexOp>(loc, memref);
    Value asI64 = builder.create<arith::IndexCastOp>(loc, builder.getI64Type(), index);
    auto ptrType = LLVM::LLVMPointerType::get(builder.getContext(), addressSpace);
    return builder.create<LLVM::IntToPtrOp>(loc, ptrType, asI64);
}

// tla.arch.sync_threads is a block-wide barrier between the threads of a SIMT
// vector function; it maps straight onto the regbase intrinsic.
static void lowerSimtSyncThreadsIn(func::FuncOp vf)
{
    SmallVector<::tla::SyncThreadsOp, 4> ops;
    vf.walk([&](::tla::SyncThreadsOp op) { ops.push_back(op); });
    for (::tla::SyncThreadsOp op : ops) {
        OpBuilder builder(op);
        builder.create<hivm_regbaseintrins::SyncThreadsOp>(op.getLoc());
        op.erase();
    }
}

// Element offset of a descriptor's view within its allocation, or a null Value
// when the view starts at the allocation's own base.
//
// The SIMT launch ABI passes a bare pointer -- there is no descriptor on the
// other side -- so a view that starts partway in has to have that start folded
// into the pointer. The formula is the one TlaLowerScalarAccessPass uses for the
// general path (coord[0]*stride[0] + coord[1]*stride[1], in elements); the two
// must agree or the same tensor means different things inside and outside a
// SIMT region.
static bool isZeroConstantValue(Value value)
{
    std::optional<int64_t> constant = getConstantIntValue(value);
    return constant && *constant == 0;
}

static FailureOr<Value> simtViewElementOffset(
    OpBuilder& builder, Location loc, const TensorDescriptor& desc, Operation* diagnosticOp)
{
    if (isZeroConstantValue(desc.coord[0]) && isZeroConstantValue(desc.coord[1]))
        return Value();

    // No contiguity requirement: linearizeSimtAccessesIn folds the tensor's real
    // strides into every index, so the pointer only has to carry the view's
    // *start*. A strided view -- tile_view of a wider parent at a non-zero
    // coordinate -- is handled by the two together: this offset moves the base,
    // and the linearized index walks the rows.

    Value rowOffset = builder.createOrFold<arith::MulIOp>(loc, desc.coord[0], desc.stride[0]);
    Value colOffset = builder.createOrFold<arith::MulIOp>(loc, desc.coord[1], desc.stride[1]);
    return builder.createOrFold<arith::AddIOp>(loc, rowOffset, colOffset);
}

// Collapse a rank-2 per-thread access into a single linear index, using the
// tensor's own strides.
//
// The SIMT launch passes one bare pointer per buffer and nothing else, so a
// memref parameter's strides are implicit and packed. A view whose rows are not
// packed -- tile_view of a wider parent, say (2,4) out of (2,8) where the rows
// are 8 apart -- cannot be described that way: memref<2x4xf32> addresses [1,2]
// as 1*4+2 instead of 1*8+2, silently reading the wrong row.
//
// Rewriting t[i, j] to flat[i*stride0 + j*stride1] removes the problem: the
// parameter stays flat and the real strides live in the arithmetic. Constant
// indices fold, so the common cases cost nothing.
static LogicalResult linearizeSimtAccessesIn(::tla::VecFuncOp vecFuncOp)
{
    // Logical dim d of a rank-R tensor lives at descriptor slot d + (2 - R): a
    // rank-1 tensor keeps its extent/stride in slot 1, a rank-2 one in slots 0
    // and 1. (The same convention TlaLowerScalarAccessPass uses.)
    auto descriptorSlot = [](size_t dim, size_t rank) { return dim + (2 - rank); };

    SmallVector<Operation*, 8> needsFolding;
    vecFuncOp.getBody().walk([&](Operation* op) {
        if (!isa<::tla::SimtLoadOp, ::tla::SimtStoreOp>(op))
            return;
        unsigned indexCount = op->getNumOperands() - (isa<::tla::SimtStoreOp>(op) ? 2 : 1);
        if (indexCount > 1) {
            needsFolding.push_back(op);
            return;
        }
        // A rank-1 view can be strided too -- every other element, say -- and its
        // single index has to be scaled or t[1] reads the neighbour. Note the
        // stride may be a *runtime* value, so this cannot be decided from the
        // static type alone: anything not provably 1 gets folded.
        if (indexCount == 1) {
            // Skip only what is *provably* unit-stride. When the descriptor is not
            // available, fall back to the static type; if that cannot prove stride 1
            // either, collect the op anyway so the loop below reports it. Silently
            // skipping here is what made a strided access read its neighbour.
            auto descOp = op->getOperand(0).getDefiningOp<::tla::TensorDescOp>();
            std::optional<int64_t> constant;
            if (descOp) {
                FailureOr<TensorDescriptor> desc = descriptorFromTensorDescOp(descOp);
                if (succeeded(desc))
                    constant = getConstantIntValue(desc->stride[1]);
            } else if (FailureOr<ParsedTensorInfo> info = parseTensorInfo(op->getOperand(0).getType());
                       succeeded(info) && !info->strides.empty() && !ShapedType::isDynamic(info->strides.back())) {
                constant = info->strides.back();
            }
            if (!constant || *constant != 1)
                needsFolding.push_back(op);
        }
    });

    // One narrowing per distinct stride value, not per access: each cast becomes a
    // captured scalar and therefore a launch argument, so emitting one per access
    // would grow the ABI linearly with the number of accesses.
    DenseMap<Value, Value> narrowedStrides;

    for (Operation* op : needsFolding) {
        bool isStore = isa<::tla::SimtStoreOp>(op);
        Value tensor = op->getOperand(0);

        // Take the strides from the descriptor rather than the type: they are SSA
        // values, so a stride computed at runtime (tla.arch.block_num() + 1, say)
        // works exactly like a constant one. The multiply lands inside the region,
        // which makes the stride a captured scalar -- the launch forwards it after
        // the buffers.
        auto descOp = tensor.getDefiningOp<::tla::TensorDescOp>();
        if (!descOp)
            return op->emitOpError(
                       "cannot linearize a per-thread access: expected a "
                       "materialized tla.tensor_desc"),
                   failure();
        FailureOr<TensorDescriptor> desc = descriptorFromTensorDescOp(descOp);
        if (failed(desc))
            return failure();

        // operands are: tensor, indices..., (value for a store)
        OperandRange indexRange = op->getOperands().drop_front(1);
        if (isStore)
            indexRange = indexRange.drop_back(1);
        SmallVector<Value, 2> indices(indexRange.begin(), indexRange.end());
        if (indices.size() != 1 && indices.size() != 2)
            return op->emitOpError("per-thread access expects one or two indices, got ") << indices.size(), failure();

        OpBuilder builder(op);
        Location loc = op->getLoc();
        Value linear;
        for (auto [dim, index] : llvm::enumerate(indices)) {
            Value stride = desc->stride[descriptorSlot(dim, indices.size())];
            Value scaled = index;
            std::optional<int64_t> constant = getConstantIntValue(stride);
            if (!constant || *constant != 1) {
                Value strideInRegion = stride;
                if (!constant) {
                    // A runtime stride crosses the launch as a captured scalar, and an
                    // index-typed launch argument reaches hivmc as an
                    // unrealized_conversion_cast and fails to translate. Narrow to i32
                    // outside the region -- that is what gets captured -- and widen back
                    // inside. NOTE: this truncates a stride of 2^31 or more; such a stride
                    // would exceed any addressable buffer on this target, but it is a
                    // silent bound rather than a checked one.
                    Value asI32 = narrowedStrides.lookup(stride);
                    if (!asI32) {
                        OpBuilder outside(vecFuncOp);
                        asI32 = outside.createOrFold<arith::IndexCastOp>(loc, outside.getI32Type(), stride);
                        narrowedStrides.insert({stride, asI32});
                    }
                    strideInRegion = builder.create<arith::IndexCastOp>(loc, builder.getIndexType(), asI32);
                }
                scaled = builder.createOrFold<arith::MulIOp>(loc, index, strideInRegion);
            }
            linear = linear ? builder.createOrFold<arith::AddIOp>(loc, linear, scaled) : scaled;
        }

        if (isStore) {
            builder.create<::tla::SimtStoreOp>(loc, tensor, ValueRange{linear}, op->getOperands().back());
        } else {
            Value loaded =
                builder.create<::tla::SimtLoadOp>(loc, op->getResult(0).getType(), tensor, ValueRange{linear});
            op->getResult(0).replaceAllUsesWith(loaded);
        }
        op->erase();
    }
    return success();
}

class LowerSimtVecFuncPattern : public OpRewritePattern<::tla::VecFuncOp> {
public:
    LowerSimtVecFuncPattern(
        MLIRContext* context, ModuleOp module, int& nextSimtRegionId, DenseMap<Value, Value>& baseMemrefCache)
        : OpRewritePattern<::tla::VecFuncOp>(context, /*benefit=*/3),
          module(module),
          nextSimtRegionId(nextSimtRegionId),
          baseMemrefCache(baseMemrefCache)
    {}

    LogicalResult matchAndRewrite(::tla::VecFuncOp vecFuncOp, PatternRewriter& rewriter) const override
    {
        if (!isSimtVecFunc(vecFuncOp))
            return rewriter.notifyMatchFailure(vecFuncOp, "not a SIMT tla.vec.func");
        if (vecFuncOp.getBody().empty())
            return rewriter.notifyMatchFailure(vecFuncOp, "expected tla.vec.func body");

        Location loc = vecFuncOp.getLoc();
        auto threadBlockDimAttr = vecFuncOp->getAttrOfType<DenseI64ArrayAttr>("thread_block_dim");
        if (!threadBlockDimAttr || threadBlockDimAttr.size() != 3)
            return vecFuncOp.emitOpError("SIMT tla.vec.func requires a 3-element 'thread_block_dim' attribute"),
                   failure();
        int64_t blockDimX = threadBlockDimAttr[0], blockDimY = threadBlockDimAttr[1], blockDimZ = threadBlockDimAttr[2];

        SimtCaptures captures;
        collectSimtCaptures(vecFuncOp, captures);
        // Runtime values computed outside the region (a core index, a block count,
        // a dynamic bound) are forwarded as extra launch arguments after the
        // buffers. Only scalars fit through that ABI.
        for (Value scalar : captures.scalars) {
            if (!scalar.getType().isIntOrIndexOrFloat())
                return vecFuncOp.emitOpError() << "a SIMT tla.vec.func can only capture scalar runtime values, "
                                                  "but this region reads one of type "
                                               << scalar.getType()
                                               << " computed outside it. Buffers must be captured as tensors; "
                                                  "anything else has no SIMT launch-argument form",
                       failure();
            // An index-typed launch argument survives TlaCompile but dies in hivmc
            // with "LLVM Translation failed for operation:
            // builtin.unrealized_conversion_cast". Integers and floats cross fine
            // (verified on device), so index is the one type that has to be narrowed
            // rather than rejected -- callers legitimately capture index-typed values
            // such as a tensor's stride.
            if (scalar.getType().isIndex())
                return vecFuncOp.emitOpError() << "a SIMT tla.vec.func cannot capture an index-typed runtime value "
                                                  "("
                                               << scalar.getType()
                                               << "): the launch ABI carries integers and floats, and an index "
                                                  "argument fails to translate in hivmc. Cast it to i32 outside the "
                                                  "region first",
                       failure();
        }

        // Base memref per captured tensor. These are the ABI buffers -- plain
        // static memrefs -- and become both the outlined parameters and, through
        // the pointer triple, the launch arguments.
        OpBuilder callBuilder(vecFuncOp);
        SmallVector<Value, 4> baseMemrefs;
        SmallVector<Type, 4> paramTypes;
        SmallVector<int, 4> addressSpaces;
        SmallVector<Value, 4> viewOffsets; // null == view starts at the base
        SmallVector<Type, 4> elementTypes;
        for (auto [operandIndex, tensor] : llvm::enumerate(captures.tensors)) {
            auto descOp = tensor.getDefiningOp<::tla::TensorDescOp>();
            if (!descOp)
                return vecFuncOp.emitOpError("SIMT capture expects a materialized tla.tensor_desc"), failure();
            FailureOr<TensorDescriptor> desc = descriptorFromTensorDescOp(descOp);
            if (failed(desc))
                return failure();

            // Address space and static-shape legality were already reported by
            // checkSimtRegionCaptures before the driver ran; bail quietly here.
            std::optional<int> addressSpace = simtAddressSpaceFor(desc->addrspace);
            if (!addressSpace)
                return failure();
            addressSpaces.push_back(*addressSpace);

            FailureOr<Value> base = getOrMaterializeDescriptorBaseMemref(
                callBuilder, loc, *desc, vecFuncOp.getOperation(), baseMemrefCache);
            if (failed(base))
                return failure();
            auto baseType = dyn_cast<MemRefType>((*base).getType());
            if (!baseType)
                return vecFuncOp.emitOpError("SIMT capture did not materialize as a memref; got ") << (*base).getType(),
                       failure();
            baseMemrefs.push_back(*base);
            elementTypes.push_back(baseType.getElementType());

            // Fold the view's start into the pointer we hand the launch, so two views
            // of one allocation stay distinguishable on the other side.
            FailureOr<Value> viewOffset = simtViewElementOffset(callBuilder, loc, *desc, vecFuncOp.getOperation());
            if (failed(viewOffset))
                return failure();
            viewOffsets.push_back(*viewOffset);
            // Parameters are always flat. One bare pointer per buffer crosses the
            // launch ABI, so a memref's implicit packed strides are the only ones it
            // could ever express; rank-2 accesses were linearized above with the
            // tensor's real strides, so the parameter only has to *span* the view --
            // the largest index that arithmetic can produce, plus one.
            FailureOr<ParsedTensorInfo> info = parseTensorInfo(tensor.getType());
            if (failed(info))
                return failure();
            size_t logicalRank = info->shape.size();
            if (logicalRank != 1 && logicalRank != 2)
                return vecFuncOp.emitOpError()
                           << "a SIMT tla.vec.func captures tensors of logical rank 1 or 2, got " << logicalRank,
                       failure();

            int64_t extent = ShapedType::kDynamic;
            if (logicalRank == 2 && info->strides.size() == 2 && !ShapedType::isDynamic(info->shape[0]) &&
                !ShapedType::isDynamic(info->shape[1]) && !ShapedType::isDynamic(info->strides[0]) &&
                !ShapedType::isDynamic(info->strides[1])) {
                // A strided view reaches past rows*cols: its last row starts at
                // (rows-1)*stride0, so bounding by rows*cols would cut the buffer short
                // and the verifier would reject the very access we just linearized.
                extent = (info->shape[0] - 1) * info->strides[0] + (info->shape[1] - 1) * info->strides[1] + 1;
            } else if (*viewOffset) {
                // The pointer now starts at the view, so the parameter must be bounded
                // by the view too: keeping the allocation's extent would let an index
                // run off the end of the buffer.
                if (std::optional<int64_t> constant = getConstantIntValue(desc->shape[1]))
                    extent = *constant;
            } else if (baseType.getRank() == 1 && !baseType.isDynamicDim(0)) {
                extent = baseType.getDimSize(0);
            } else if (std::optional<int64_t> constant = getConstantIntValue(desc->shape[1])) {
                extent = *constant;
            }
            if (ShapedType::isDynamic(extent))
                return failure();
            SmallVector<int64_t, 1> paramShape{extent};
            (void)operandIndex;
            // The parameter must carry an *integer* address space, not
            // #hivm.address_space<...>: only then does memref-to-llvm lower it to a
            // bare pointer matching the !llvm.ptr the launch passes. With the hivm
            // attribute it becomes a descriptor struct and hivmc-a5 aborts with
            // "Calling a function with a bad signature!".
            paramTypes.push_back(MemRefType::get(
                paramShape, baseType.getElementType(), MemRefLayoutAttrInterface{},
                rewriter.getI64IntegerAttr(*addressSpace)));
        }

        // ---- the outlined vector function ----
        auto parentFunc = vecFuncOp->getParentOfType<func::FuncOp>();
        std::string vfName = (parentFunc ? parentFunc.getName().str() : std::string("kernel")) + "_vf_simt" +
                             (nextSimtRegionId ? std::to_string(nextSimtRegionId) : std::string());
        ++nextSimtRegionId;

        for (Value scalar : captures.scalars)
            paramTypes.push_back(scalar.getType());

        bool hasPrint = false;
        vecFuncOp.walk([&](Operation* op) { hasPrint |= isa<DebugPrintOp, PrintTensorOp>(op); });
        Value printWorkspace;
        if (hasPrint) {
            printWorkspace = getOrAppendSimtPrintWorkspaceArg(parentFunc);
            paramTypes.push_back(callBuilder.getI64Type());
        }
        ModuleOp mod = module;
        OpBuilder moduleBuilder(mod.getBodyRegion());
        moduleBuilder.setInsertionPointToEnd(mod.getBody());
        auto vfType = FunctionType::get(rewriter.getContext(), paramTypes, {});
        auto vf = moduleBuilder.create<func::FuncOp>(loc, vfName, vfType);
        MLIRContext* ctx = rewriter.getContext();
        vf->setAttr(
            hivm_regbaseintrins::kDavinciCallingConvAttrName,
            hivm_regbaseintrins::SIMT_EntryAttr::get(ctx, static_cast<uint32_t>(blockDimX * blockDimY * blockDimZ)));
        vf->setAttr("noinline", BoolAttr::get(ctx, false));
        // SIMT regions are always vector-core functions. A pure AIV module gets
        // away without this attribute because hivm.module_core_type covers the
        // whole module, but a mixed AIC+AIV module has no single core type, and
        // hivmc-a5 rejects the outlined function with "Unknown core type". SIMT
        // is inherently vector-core work, so mark it AIV outright rather than
        // deriving it from the enclosing function or the pass order.
        vf->setAttr(hivm::TFuncCoreTypeAttr::name, hivm::TFuncCoreTypeAttr::get(ctx, hivm::TFuncCoreType::AIV));

        Block* vfBody = vf.addEntryBlock();
        OpBuilder bodyBuilder(vfBody, vfBody->begin());
        IRMapping mapping;
        ArrayRef<BlockArgument> vfArgs = vfBody->getArguments();
        for (auto [tensor, arg] : llvm::zip_equal(captures.tensors, vfArgs.take_front(captures.tensors.size())))
            mapping.map(tensor, arg);
        for (auto [scalar, arg] :
             llvm::zip_equal(captures.scalars, vfArgs.slice(captures.tensors.size(), captures.scalars.size())))
            mapping.map(scalar, arg);
        for (Operation* constant : captures.constants)
            bodyBuilder.clone(*constant, mapping);

        // Clone the region body in. Tensor operands are remapped to the memref
        // parameters as we go, which leaves the cloned scalar accesses momentarily
        // holding a memref where a !tla.tensor is declared; the walk below converts
        // them to memref.load/store, at whatever nesting depth they sit.
        for (Operation& op : vecFuncOp.getBody().front())
            bodyBuilder.clone(op, mapping);
        bodyBuilder.create<func::ReturnOp>(loc);

        if (hasPrint && failed(lowerSimtPrints(vf, vfArgs.back())))
            return failure();
        if (failed(rewriteSimtAccessesOntoParams(vf)))
            return failure();
        lowerSimtTripleOpsIn<
            ::tla::ThreadIdxOp, hivm_regbaseintrins::ThreadIdXOp, hivm_regbaseintrins::ThreadIdYOp,
            hivm_regbaseintrins::ThreadIdZOp>(vf);
        lowerSimtTripleOpsIn<
            ::tla::ThreadBlockDimOp, hivm_regbaseintrins::BlockDimXOp, hivm_regbaseintrins::BlockDimYOp,
            hivm_regbaseintrins::BlockDimZOp>(vf);
        lowerSimtArithmeticIn(vf);
        lowerSimtSyncThreadsIn(vf);

        // ---- the launch ----
        SmallVector<Value, 4> launchArgs;
        for (auto [base, addressSpace, viewOffset, elementType] :
             llvm::zip_equal(baseMemrefs, addressSpaces, viewOffsets, elementTypes)) {
            Value pointer = memrefToRawPointer(callBuilder, loc, base, addressSpace);
            if (viewOffset) {
                Value index = callBuilder.createOrFold<arith::IndexCastOp>(loc, callBuilder.getI64Type(), viewOffset);
                auto pointerType = LLVM::LLVMPointerType::get(callBuilder.getContext(), addressSpace);
                pointer = callBuilder.create<LLVM::GEPOp>(loc, pointerType, elementType, pointer, ValueRange{index});
            }
            launchArgs.push_back(pointer);
        }
        // Captured runtime scalars follow the buffers, in capture order: the ABI is
        // positional, so this must match the parameter list built above.
        for (Value scalar : captures.scalars)
            launchArgs.push_back(scalar);
        if (hasPrint)
            launchArgs.push_back(printWorkspace);
        auto i64 = callBuilder.getI64Type();
        Value tx = callBuilder.create<arith::ConstantOp>(loc, i64, callBuilder.getI64IntegerAttr(blockDimX));
        Value ty = callBuilder.create<arith::ConstantOp>(loc, i64, callBuilder.getI64IntegerAttr(blockDimY));
        Value tz = callBuilder.create<arith::ConstantOp>(loc, i64, callBuilder.getI64IntegerAttr(blockDimZ));
        callBuilder.create<hivm_regbaseintrins::LaunchFuncOp>(
            loc, FlatSymbolRefAttr::get(ctx, vfName), tx, ty, tz, launchArgs);

        rewriter.eraseOp(vecFuncOp);
        return success();
    }

private:
    // Convert every cloned tla.simt_load/simt_store onto the memref parameter its
    // tensor operand was mapped to.
    static LogicalResult rewriteSimtAccessesOntoParams(func::FuncOp vf)
    {
        SmallVector<Operation*, 8> stale;
        vf.walk([&](Operation* op) {
            if (isa<::tla::SimtLoadOp, ::tla::SimtStoreOp>(op))
                stale.push_back(op);
        });
        for (Operation* op : stale) {
            OpBuilder builder(op);
            // The clone remapped the tensor operand to a memref parameter, so the op
            // no longer matches its own declared operand type. Read that operand
            // positionally: the generated getSource()/getDest() accessors cast to
            // TypedValue<TlaTensorType>, which asserts in an assertions-enabled build
            // (and silently returned a mistyped value in one without).
            if (auto load = dyn_cast<::tla::SimtLoadOp>(op)) {
                Value source = op->getOperand(0);
                SmallVector<Value, 2> indices(load.getIndices().begin(), load.getIndices().end());
                if (!isa<MemRefType>(source.getType()))
                    return load.emitOpError("tla.simt_load source did not map to a memref parameter"), failure();
                Value loaded = builder.create<mlir::memref::LoadOp>(load.getLoc(), source, indices);
                load.getResult().replaceAllUsesWith(loaded);
            } else {
                auto store = cast<::tla::SimtStoreOp>(op);
                Value dest = op->getOperand(0);
                SmallVector<Value, 2> indices(store.getIndices().begin(), store.getIndices().end());
                if (!isa<MemRefType>(dest.getType()))
                    return store.emitOpError("tla.simt_store dest did not map to a memref parameter"), failure();
                builder.create<mlir::memref::StoreOp>(store.getLoc(), store.getValue(), dest, indices);
            }
            op->erase();
        }
        return success();
    }

    ModuleOp module;
    int& nextSimtRegionId;
    DenseMap<Value, Value>& baseMemrefCache;
};

static LogicalResult linearizeSimtAccesses(func::FuncOp funcOp)
{
    LogicalResult result = success();
    funcOp.walk([&](::tla::VecFuncOp vecFuncOp) {
        if (!isSimtVecFunc(vecFuncOp))
            return;
        if (failed(linearizeSimtAccessesIn(vecFuncOp)))
            result = failure();
    });
    return result;
}

static LogicalResult checkSimtRegionCaptures(func::FuncOp funcOp)
{
    LogicalResult result = success();
    funcOp.walk([&](::tla::VecFuncOp vecFuncOp) {
        if (!isSimtVecFunc(vecFuncOp))
            return;
        SimtCaptures captures;
        collectSimtCaptures(vecFuncOp, captures);

        // Gather every offender first: one diagnostic per region listing them all
        // beats one per tensor, especially since the frontend runs the pipeline
        // twice (typed bridge, then CLI fallback) and doubles whatever is emitted.
        SmallVector<std::string, 4> nonGm;
        SmallVector<unsigned, 4> dynamic;
        for (auto [operandIndex, tensor] : llvm::enumerate(captures.tensors)) {
            auto descOp = tensor.getDefiningOp<::tla::TensorDescOp>();
            if (!descOp)
                continue;
            FailureOr<TensorDescriptor> desc = descriptorFromTensorDescOp(descOp);
            if (failed(desc))
                continue;
            if (!simtAddressSpaceFor(desc->addrspace))
                nonGm.push_back((Twine("#") + Twine(operandIndex + 1) + " in '" + desc->addrspace + "'").str());
            else if (!getConstantIntValue(desc->shape[1]))
                dynamic.push_back(operandIndex + 1);
        }

        auto listTensors = [&](ArrayRef<unsigned> indices) {
            std::string text;
            llvm::raw_string_ostream os(text);
            llvm::interleaveComma(indices, os, [&](unsigned index) { os << "#" << index; });
            return os.str();
        };

        // A SIMT vector function is launched with GM pointers only. A UB operand
        // never gets a base established and silently reads as zeros, so refuse it
        // rather than let the kernel produce wrong answers.
        if (!nonGm.empty()) {
            vecFuncOp.emitOpError() << "a SIMT tla.vec.func can only address GM or UB, but " << nonGm.size() << " of "
                                    << captures.tensors.size()
                                    << " tensor(s) used by this region are not: " << llvm::join(nonGm, ", ")
                                    << ". Buffers reach an outlined SIMT vector function as raw pointers, and "
                                       "only the GM and UB address spaces have a SIMT pointer form";
            result = failure();
        }
        // One pointer per buffer crosses the launch ABI, so the parameter must be a
        // statically shaped memref: a dynamic extent lowers to a 5-field descriptor
        // and hivmc-a5 aborts with an opaque LLVM assertion.
        if (!dynamic.empty()) {
            vecFuncOp.emitOpError() << "a SIMT tla.vec.func requires statically shaped buffers, but tensor(s) "
                                    << listTensors(dynamic) << " of " << captures.tensors.size()
                                    << " used by this region have a dynamic extent. The launch passes one "
                                       "pointer per buffer, while a dynamic memref lowers to a 5-field "
                                       "descriptor (base, aligned, offset, size, stride), which makes hivmc-a5 "
                                       "fail with \"Calling a function with a bad signature!\". Drop "
                                       "mark_compact_shape_dynamic(...) from the host tensor so its extent is "
                                       "known at compile time";
            result = failure();
        }
    });
    return result;
}

} // namespace

void populateSimtVectorPatterns(
    RewritePatternSet& patterns, ModuleOp module, int& nextSimtRegionId, DenseMap<Value, Value>& loweredMemrefByValue)
{
    MLIRContext* ctx = patterns.getContext();
    patterns.add<LowerSimtVecFuncPattern>(ctx, module, nextSimtRegionId, loweredMemrefByValue);
    patterns.add<LowerThreadIdxPattern, LowerBlockDimPattern>(ctx);
    populateTlaCopyPatterns(patterns, loweredMemrefByValue);
}

LogicalResult runSimtPreChecks(func::FuncOp funcOp)
{
    if (failed(checkSimtRegionCaptures(funcOp)))
        return failure();
    return linearizeSimtAccesses(funcOp);
}

namespace {

/// Lower the SIMT half of the tla.vector lowering: a `tla.vec.func(mode="simt")`
/// describes per-thread scalar work, which is a different machine model from the
/// whole-vector AVE instructions the SIMD pass emits. The two share no code.
///
/// Runs before tla-vector-region-simd.
class TlaVectorRegionSimtPass : public PassWrapper<TlaVectorRegionSimtPass, OperationPass<ModuleOp>> {
public:
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TlaVectorRegionSimtPass)

    StringRef getArgument() const override
    {
        return "tla-vector-region-simt";
    }
    StringRef getName() const override
    {
        return "TlaVectorRegionSimtPass";
    }
    StringRef getDescription() const override
    {
        return "Outline tla.vec.func(mode=simt) regions and lower them to per-thread regbase IR.";
    }

    void getDependentDialects(DialectRegistry& registry) const override
    {
        registry.insert<
            arith::ArithDialect, func::FuncDialect, mlir::memref::MemRefDialect, hivm::HIVMDialect,
            vector::VectorDialect, hivm_regbaseintrins::HIVMRegbaseIntrinsDialect, LLVM::LLVMDialect, math::MathDialect,
            ::tla::TlaDialect>();
    }

    void runOnOperation() override
    {
        ModuleOp module = getOperation();
        nextSimtRegionId = 0;

        // Snapshot up front: lowering a vec.func appends an outlined helper to the
        // module, and that helper must not be fed back through the driver.
        SmallVector<func::FuncOp, 4> funcOps(module.getOps<func::FuncOp>());
        for (func::FuncOp funcOp : funcOps) {
            if (funcOp.isDeclaration())
                continue;
            if (funcOp->hasAttr(kHivmVectorFunctionAttrName))
                continue;
            std::optional<hivm::TFuncCoreType> coreType = getFunctionCoreType(funcOp.getOperation());
            if (coreType != hivm::TFuncCoreType::AIV && coreType != hivm::TFuncCoreType::MIX)
                continue;
            // Mode-agnostic validation, run once for the whole lowering: the arch
            // ops it rejects are illegal inside a vec.func of either mode.
            if (failed(checkNoArchOpsInVecFunc(funcOp))) {
                signalPassFailure();
                return;
            }
            if (failed(runSimtPreChecks(funcOp))) {
                signalPassFailure();
                return;
            }
            inlineVectorRegionWrappers(funcOp);
            ::tla::TlaTensorMemrefLowering lowering;
            RewritePatternSet patterns(&getContext());
            patterns.add<InlineVectorRegionWrapperPattern>(&getContext());
            populateSimtVectorPatterns(patterns, module, nextSimtRegionId, lowering.loweredMemrefByValue);
            // Both vector passes run the greedy driver over every AIV/MIX function,
            // whether or not they own anything in it -- see the note on the SIMD
            // pass for why neither one can be gated. Two unconditional driver runs
            // only compose if neither reorders the other's constant pool, so both
            // disable constant CSE and leave it to the pipeline's later CSEPass.
            GreedyRewriteConfig simtCfg;
            simtCfg.cseConstants = false;
            if (failed(mlir::applyPatternsGreedily(funcOp, std::move(patterns), simtCfg))) {
                signalPassFailure();
                return;
            }
        }

        // A launched SIMT vector function makes the entry a regbase SIMT kernel as
        // well as a SIMD one; the two sets of entry attributes coexist.
        if (nextSimtRegionId > 0) {
            MLIRContext* ctx = &getContext();
            StringRef entryAttr = hacc::stringifyEnum(hacc::HACCToLLVMIRTranslateAttr::ENTRY);
            for (func::FuncOp funcOp : module.getOps<func::FuncOp>()) {
                if (!funcOp->hasAttr(entryAttr))
                    continue;
                funcOp->setAttr(hivm_regbaseintrins::kDavinciKernelAttrName, UnitAttr::get(ctx));
                setC310RegbaseTargetAttr(funcOp.getOperation(), ctx);
            }
        }
    }

private:
    int nextSimtRegionId = 0;
};

} // namespace

std::unique_ptr<Pass> createTlaVectorRegionSimtPass()
{
    return std::make_unique<TlaVectorRegionSimtPass>();
}

void registerTlaVectorRegionSimtPass()
{
    PassRegistration<TlaVectorRegionSimtPass>();
}

} // namespace tla
