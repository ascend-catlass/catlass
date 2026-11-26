#include "Passes/TlaDebugPrint.h"
#include "PassesCommon.h"
#include "PassesInternal.h"
#include "TlaSimtWarpLowering.h"
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
    // Memrefs a raw-pointer chain (t.ptr on a memref-backed tensor) reads from
    // inside the region. Each crosses as its own bare-pointer launch argument,
    // with no view offset folded in -- t.ptr is the raw storage base.
    SmallVector<Value, 4> pointerBuffers;
    // Memrefs the region indexes directly instead of going through a tensor.
    // The launch ABI has no layout to carry, so these are rejected rather than
    // captured under an identity layout that would silently misaddress a
    // strided or dynamic-stride buffer.
    SmallVector<Value, 4> directMemrefs;
};

static bool isDefinedOutside(Value value, Region& region)
{
    Region* defRegion = value.getParentRegion();
    return !defRegion || !region.isAncestor(defRegion);
}

// tla-lower-ptr folds pointer arithmetic to i64 and materializes the typed
// pointer back with tla.inttoptr at each use. For an atomic whose address was
// computed outside the region, that materialization can land outside too; a
// !tla.ptr cannot cross the boundary, but its i64 address can. Rebuild the
// pointer at the atomic: the i64 becomes an ordinary scalar capture and the
// inttoptr is recreated inside. Pointers that are not backed by an i64 address
// remain rejected, which keeps the "build the pointer inside the region" API
// rule where provenance is still visible.
static void rebuildOutsideAtomicPtrsInside(::tla::VecFuncOp vecFuncOp)
{
    Region& region = vecFuncOp.getBody();
    SmallVector<::tla::SimtAtomicAddOp, 8> atomics;
    region.walk([&](::tla::SimtAtomicAddOp op) { atomics.push_back(op); });
    for (::tla::SimtAtomicAddOp op : atomics) {
        Value pointer = op.getPtr();
        if (!isDefinedOutside(pointer, region))
            continue;
        auto intToPtr = pointer.getDefiningOp<::tla::IntToPtrOp>();
        if (!intToPtr)
            continue;
        OpBuilder builder(op);
        Value rebuilt = builder.create<::tla::IntToPtrOp>(op.getLoc(), pointer.getType(), intToPtr.getAddr());
        op->setOperand(0, rebuilt);
    }
}

static void collectSimtCaptures(::tla::VecFuncOp vecFuncOp, SimtCaptures& captures)
{
    Region& region = vecFuncOp.getBody();
    SetVector<Value> tensorSet;
    SetVector<Value> scalarSet;
    SetVector<Operation*> constantSet;
    SetVector<Value> pointerSet;
    SetVector<Value> directMemrefSet;

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
        } else if (auto ptrWithCoord = dyn_cast<::tla::TensorPtrWithCoordOp>(op)) {
            // The view-start address reads off the outlined buffer parameter,
            // so the source tensor must be captured even if nothing else in
            // the region touches it.
            if (isDefinedOutside(ptrWithCoord.getSrc(), region))
                tensorSet.insert(ptrWithCoord.getSrc());
        }
        for (Value operand : op->getOperands()) {
            if (!isDefinedOutside(operand, region))
                continue;
            if (tensorSet.contains(operand) || isa<::tla::TlaTensorType>(operand.getType()))
                continue;
            // A raw-pointer chain (t.ptr on a memref-backed tensor) reads just
            // its storage address: tla-lower-ptr expressed that as
            // extract_aligned_pointer_as_index. Such a memref crosses as its own
            // bare-pointer launch argument, and the body never indexes it, so
            // its layout cannot be misread. Any other use indexes the memref, and
            // the ABI has no layout to carry -- reject it instead of dropping
            // layout/stride.
            if (isa<MemRefType>(operand.getType())) {
                if (isa<mlir::memref::ExtractAlignedPointerAsIndexOp>(op))
                    pointerSet.insert(operand);
                else
                    directMemrefSet.insert(operand);
                continue;
            }
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
    captures.pointerBuffers.assign(pointerSet.begin(), pointerSet.end());
    captures.directMemrefs.assign(directMemrefSet.begin(), directMemrefSet.end());
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

// Integer address space for a memref's memory space, or failure for one a SIMT
// vector function cannot reach. Kernel-argument memrefs carry the
// #hivm.address_space attribute; the integer forms are what memref-to-llvm
// lowers to a bare pointer matching the launch's !llvm.ptr.
static std::optional<int> simtAddressSpaceForMemref(MemRefType type)
{
    Attribute space = type.getMemorySpace();
    if (auto hivmSpace = dyn_cast_or_null<hivm::AddressSpaceAttr>(space)) {
        if (hivmSpace.getAddressSpace() == hivm::AddressSpace::GM)
            return kSimtGmAddressSpace;
        if (hivmSpace.getAddressSpace() == hivm::AddressSpace::UB)
            return kSimtUbAddressSpace;
        return std::nullopt;
    }
    if (auto intSpace = dyn_cast_or_null<IntegerAttr>(space)) {
        int64_t value = intSpace.getInt();
        if (value == kSimtGmAddressSpace || value == kSimtUbAddressSpace)
            return static_cast<int>(value);
    }
    return std::nullopt;
}

// Address-space name for a memref capture's diagnostic. Buffer captures print
// the tensor's own ``gm`` / ``l1`` string; a raw-pointer capture carries the
// same information on the memref's memory space, so spell it the same way
// instead of leaking the HIVM enum name (``cbuf`` for L1).
static std::string memrefAddressSpaceName(MemRefType type)
{
    Attribute space = type.getMemorySpace();
    if (auto hivmSpace = dyn_cast_or_null<hivm::AddressSpaceAttr>(space)) {
        switch (hivmSpace.getAddressSpace()) {
            case hivm::AddressSpace::Zero:
                return "generic";
            case hivm::AddressSpace::GM:
                return "gm";
            case hivm::AddressSpace::L1:
                return "l1";
            case hivm::AddressSpace::L0A:
                return "l0a";
            case hivm::AddressSpace::L0B:
                return "l0b";
            case hivm::AddressSpace::L0C:
                return "l0c";
            case hivm::AddressSpace::UB:
                return "ub";
            case hivm::AddressSpace::BiasBUF:
                return "biasbuf";
        }
        return hivm::stringifyAddressSpace(hivmSpace.getAddressSpace()).str();
    }
    if (auto intSpace = dyn_cast_or_null<IntegerAttr>(space)) {
        switch (intSpace.getInt()) {
            case 0:
                return "generic";
            case 1:
                return "gm";
            case 2:
                return "l1";
            case 3:
                return "l0a";
            case 4:
                return "l0b";
            case 5:
                return "l0c";
            case 6:
                return "ub";
            case 12:
                return "biasbuf";
            default:
                break;
        }
        return std::to_string(intSpace.getInt());
    }
    return "default";
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

// The suffix the atom unit's type takes in the intrinsic name. Unsigned adds use
// the signed form: addition is bit-identical, and the atom unit has no separate
// unsigned add.
static std::optional<StringRef> simtAtomicAddTypeToken(Type elementType)
{
    if (auto integer = dyn_cast<IntegerType>(elementType)) {
        if (integer.getWidth() == 32)
            return "s32";
        if (integer.getWidth() == 64)
            return "s64";
        return std::nullopt;
    }
    if (elementType.isF32())
        return "fp32";
    if (elementType.isF16())
        return "fp16";
    if (elementType.isBF16())
        return "bf16";
    return std::nullopt;
}

// A per-thread atomic add becomes llvm.hivm.atom.ADD.<G|S>.<type> at the address
// the caller computed. The address is rebuilt here rather than carried across the
// launch: a !tla.ptr cannot cross the region boundary, so the chain's root is
// either a captured tensor (tensor_ptr, whose memref parameter is the very
// pointer the launch passes) or a captured integer address (inttoptr). The
// offsets between the root and the atomic are folded into one byte offset, each
// ptr_add scaled by the pointee type it was written against.
//
// The atom instruction is fetching, so the old value rides the intrinsic result;
// for f16/bf16 that hardware value is not the original element, and the frontend
// drops it for those types.
static LogicalResult lowerSimtAtomicAdd(OpBuilder& builder, ::tla::SimtAtomicAddOp op)
{
    Location loc = op.getLoc();
    auto ptrType = dyn_cast<::tla::PtrType>(op.getPtr().getType());
    if (!ptrType)
        return op.emitOpError("tla.simt_atomic_add pointer did not survive as !tla.ptr"), failure();

    int addressSpace;
    switch (ptrType.getAddrspace()) {
        case AddressSpace::gm:
            addressSpace = kSimtGmAddressSpace;
            break;
        case AddressSpace::ub:
            addressSpace = kSimtUbAddressSpace;
            break;
        default:
            return op.emitOpError("tla.simt_atomic_add pointer must be in gm or ub, got ")
                       << stringifyAddressSpace(ptrType.getAddrspace()),
                   failure();
    }

    Type elementType = ptrType.getPointee();
    std::optional<StringRef> typeToken = simtAtomicAddTypeToken(elementType);
    if (!typeToken)
        return op.emitOpError() << "unsupported atomic add element type " << elementType, failure();
    int64_t elementBytes = getByteSizeOfFixedWidthScalarType(elementType);
    if (elementBytes <= 0)
        return op.emitOpError() << "atomic add element type " << elementType << " has no fixed byte width", failure();

    // The LLVM dialect has no unsigned integers; carry the bits in the signless
    // storage type. arith.bitcast rejects unsigned operands, so this is the same
    // unrealized cast the unsigned scalar load/store path uses.
    Type storageType = elementType;
    if (auto integer = dyn_cast<IntegerType>(elementType))
        storageType = builder.getIntegerType(integer.getWidth());
    Value value = op.getValue();
    if (value.getType() != storageType)
        value = builder.create<mlir::UnrealizedConversionCastOp>(loc, TypeRange{storageType}, ValueRange{value})
                    .getResult(0);

    // Walk the chain back to its root, summing the offsets on the way as bytes:
    // a recast in the middle changes the element unit, so each ptr_add is scaled
    // by the pointee type it was written against.
    Value base = op.getPtr();
    Value byteOffset;
    while (true) {
        if (auto add = base.getDefiningOp<::tla::PtrAddOp>()) {
            int64_t addBytes = elementBytes;
            if (auto addPtrType = dyn_cast<::tla::PtrType>(add.getPtr().getType()))
                addBytes = getByteSizeOfFixedWidthScalarType(addPtrType.getPointee());
            if (addBytes <= 0)
                return op.emitOpError("tla.ptr_add pointee has no fixed byte width"), failure();
            Value step = castValueToI64(builder, loc, add.getOffset());
            if (addBytes != 1) {
                Value scale = builder.create<arith::ConstantIntOp>(loc, addBytes, 64);
                step = builder.createOrFold<arith::MulIOp>(loc, step, scale);
            }
            byteOffset = byteOffset ? builder.createOrFold<arith::AddIOp>(loc, byteOffset, step) : step;
            base = add.getPtr();
            continue;
        }
        if (auto recast = base.getDefiningOp<::tla::RecastPtrOp>()) {
            base = recast.getSrc();
            continue;
        }
        break;
    }

    auto pointerType = LLVM::LLVMPointerType::get(builder.getContext(), addressSpace);
    Value pointer;
    if (auto ptrWithCoord = base.getDefiningOp<::tla::TensorPtrWithCoordOp>()) {
        // ptr_with_coord is the view-start address. The source tensor was
        // captured as a buffer parameter whose launch pointer already carries
        // the view start, so read that parameter directly.
        Value param = ptrWithCoord->getOperand(0);
        if (!isa<MemRefType>(param.getType()))
            return op.emitOpError("tla.simt_atomic_add ptr_with_coord source did not map to a memref parameter"),
                   failure();
        pointer = memrefToRawPointer(builder, loc, param, addressSpace);
    } else if (auto intToPtr = base.getDefiningOp<::tla::IntToPtrOp>()) {
        pointer = builder.create<LLVM::IntToPtrOp>(loc, pointerType, intToPtr.getAddr());
    } else {
        return op.emitOpError(
                   "tla.simt_atomic_add pointer must come from tla.tensor_ptr_with_coord or "
                   "tla.inttoptr inside the region"),
               failure();
    }

    if (byteOffset)
        pointer = builder.create<LLVM::GEPOp>(loc, pointerType, builder.getI8Type(), pointer, ValueRange{byteOffset});
    // 0 is L2_CACHE_HINT_NORMAL_FV, the cache hint the DPX lowering uses.
    Value cacheHint = builder.create<arith::ConstantOp>(loc, builder.getI32IntegerAttr(0));
    std::string intrinsic =
        ("llvm.hivm.atom.ADD." + Twine(addressSpace == kSimtGmAddressSpace ? "G" : "S") + "." + *typeToken).str();
    Value oldValue = builder
                         .create<LLVM::CallIntrinsicOp>(
                             loc, storageType, builder.getStringAttr(intrinsic), ValueRange{pointer, value, cacheHint})
                         .getResult(0);
    // The intrinsic returns the signless storage type; consumers still expect the
    // declared pointee, so unsigned results need the bridge back, as in the
    // scalar atomic lowering.
    if (oldValue.getType() != elementType)
        oldValue = builder.create<mlir::UnrealizedConversionCastOp>(loc, TypeRange{elementType}, ValueRange{oldValue})
                       .getResult(0);
    op.getResult().replaceAllUsesWith(oldValue);
    return success();
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

// tla.arch.shuffle reads a register from another lane of the same warp.
//
// Every mode and type goes through one 32-bit control word. The field layout is
// fixed by the hardware and is the same one AscendDPXToHIVMRegbaseIntrins builds
// for the Triton path:
//
//   ctrl[4:0]   source lane / delta / lane mask
//   ctrl[12:8]  clamp      -- the group's upper lane bound
//   ctrl[20:16] lane mask  -- the group's lower lane bound, as 32 - width
//
// `up` clamps at 0 because it walks towards lane 0 and must stop there; the
// other three clamp at 0x1f, the top of the warp.
static Value buildShuffleCtrl(OpBuilder& builder, Location loc, ::ShuffleMode mode, Value index, int64_t width)
{
    const int64_t clamp = (mode == ::ShuffleMode::up) ? 0 : 0x1f;
    const int64_t laneMask = 32 - width;
    // clamp and laneMask come from the width attribute, so everything but the
    // index is a compile-time constant and folds into a single immediate.
    const int64_t constantPart = (clamp << 8) | (laneMask << 16);

    Type i32 = builder.getI32Type();
    Value masked = builder.create<arith::AndIOp>(
        loc, index, builder.create<arith::ConstantOp>(loc, builder.getI32IntegerAttr(0x1f)));
    return builder.create<arith::OrIOp>(
        loc, masked, builder.create<arith::ConstantOp>(loc, builder.getI32IntegerAttr(constantPart)));
}

// Create the intrinsic for one mode and one natively supported type.
template <typename IdxOpTy, typename UpOpTy, typename DownOpTy, typename BflyOpTy>
static Value buildShuffleIntrinsic(
    OpBuilder& builder, Location loc, ::ShuffleMode mode, Type type, Value value, Value ctrl)
{
    switch (mode) {
        case ::ShuffleMode::idx:
            return builder.create<IdxOpTy>(loc, type, value, ctrl).getRes();
        case ::ShuffleMode::up:
            return builder.create<UpOpTy>(loc, type, value, ctrl).getRes();
        case ::ShuffleMode::down:
            return builder.create<DownOpTy>(loc, type, value, ctrl).getRes();
        case ::ShuffleMode::bfly:
            return builder.create<BflyOpTy>(loc, type, value, ctrl).getRes();
    }
    llvm_unreachable("unhandled shuffle mode");
}

// Dispatch on the value type. i32/i64/f32/f16 have their own intrinsic; i8, i16
// and bf16 do not, so they ride the i32 one. That is exact: a shuffle copies a
// register between lanes without interpreting it, so widening the bit pattern
// and narrowing it back cannot change the value.
static Value buildShuffle(OpBuilder& builder, Location loc, ::ShuffleMode mode, Value value, Value index, int64_t width)
{
    namespace rb = hivm_regbaseintrins;
    Value ctrl = buildShuffleCtrl(builder, loc, mode, index, width);
    Type type = value.getType();

    if (type.isInteger(32))
        return buildShuffleIntrinsic<rb::ShflIdxOpI32, rb::ShflUpOpI32, rb::ShflDownOpI32, rb::ShflBflyOpI32>(
            builder, loc, mode, type, value, ctrl);
    if (type.isInteger(64))
        return buildShuffleIntrinsic<rb::ShflIdxOpI64, rb::ShflUpOpI64, rb::ShflDownOpI64, rb::ShflBflyOpI64>(
            builder, loc, mode, type, value, ctrl);
    if (type.isF32())
        return buildShuffleIntrinsic<rb::ShflIdxOpF32, rb::ShflUpOpF32, rb::ShflDownOpF32, rb::ShflBflyOpF32>(
            builder, loc, mode, type, value, ctrl);
    if (type.isF16())
        return buildShuffleIntrinsic<rb::ShflIdxOpF16, rb::ShflUpOpF16, rb::ShflDownOpF16, rb::ShflBflyOpF16>(
            builder, loc, mode, type, value, ctrl);

    // Narrow types: bitcast anything non-integral to an integer of the same
    // width first, zero-extend to i32, shuffle, then undo both steps.
    Type i32 = builder.getI32Type();
    const unsigned bitWidth = type.getIntOrFloatBitWidth();
    Type narrowInt = builder.getIntegerType(bitWidth);
    Value bits = type.isIntOrIndex() ? value : builder.create<arith::BitcastOp>(loc, narrowInt, value).getResult();
    Value widened = builder.create<arith::ExtUIOp>(loc, i32, bits);
    Value shuffled = buildShuffleIntrinsic<rb::ShflIdxOpI32, rb::ShflUpOpI32, rb::ShflDownOpI32, rb::ShflBflyOpI32>(
        builder, loc, mode, i32, widened, ctrl);
    Value narrowed = builder.create<arith::TruncIOp>(loc, narrowInt, shuffled);
    return type.isIntOrIndex() ? narrowed : builder.create<arith::BitcastOp>(loc, type, narrowed).getResult();
}

static void lowerSimtShuffleIn(func::FuncOp vf)
{
    SmallVector<::tla::ArchShuffleOp, 4> ops;
    vf.walk([&](::tla::ArchShuffleOp op) { ops.push_back(op); });
    for (::tla::ArchShuffleOp op : ops) {
        OpBuilder builder(op);
        Value lowered = buildShuffle(builder, op.getLoc(), op.getMode(), op.getValue(), op.getIndex(), op.getWidth());
        op.getResult().replaceAllUsesWith(lowered);
        op.erase();
    }
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
        // Operands are tensor, indices..., then a trailing value for the ops that
        // write; a load has no trailing operand.
        unsigned indexCount = op->getNumOperands() - (isa<::tla::SimtLoadOp>(op) ? 1 : 2);
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
        bool writes = isa<::tla::SimtStoreOp>(op);
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

        // operands are: tensor, indices..., (value for a store/atomic add)
        OperandRange indexRange = op->getOperands().drop_front(1);
        if (writes)
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

        if (isa<::tla::SimtStoreOp>(op)) {
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

        // A pointer folded to i64 by tla-lower-ptr may be materialized outside
        // the region; rebuild it at the atomic before capture analysis so the
        // i64 crosses as an ordinary scalar.
        rebuildOutsideAtomicPtrsInside(vecFuncOp);

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
        // Logical elements per storage element of the buffer parameter: 2 for a
        // packed 4-bit view buffered as i8, 1 otherwise.
        SmallVector<int64_t, 4> elementsPerStorageUnit;
        for (Value tensor : captures.tensors) {
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
            // of one allocation stay distinguishable on the other side. The SIMT
            // launch ABI passes one bare pointer -- there is no descriptor on the
            // other side -- so the start has to move into that pointer. The formula
            // is the one TlaLowerScalarAccessPass and tla.tensor_ptr_with_coord use
            // (coord[0]*stride[0] + coord[1]*stride[1], in elements), or the same
            // tensor would mean different addresses inside and outside the region.
            Value viewOffset = descriptorViewElementOffset(callBuilder, loc, *desc);
            viewOffsets.push_back(viewOffset);
            // A packed sub-byte element type (fp4) is buffered as bytes, so the
            // launch GEP steps in storage elements: two logical elements share
            // one byte. The view start must land on a byte boundary; a static
            // odd offset is a half-byte address and is diagnosed here.
            int64_t logicalBits = elementStorageBits(desc->elementType);
            if (logicalBits <= 0)
                logicalBits = getBitSizeOfFixedWidthScalarType(desc->elementType);
            int64_t storageBits = elementStorageBits(baseType.getElementType());
            if (storageBits <= 0)
                storageBits = getBitSizeOfFixedWidthScalarType(baseType.getElementType());
            int64_t storageElementsPerLogicalElement =
                (logicalBits > 0 && storageBits > logicalBits && storageBits % logicalBits == 0) ?
                    storageBits / logicalBits :
                    1;
            if (storageElementsPerLogicalElement > 1) {
                if (std::optional<int64_t> constant = getConstantIntValue(viewOffset);
                    constant && *constant % storageElementsPerLogicalElement != 0)
                    return vecFuncOp.emitOpError()
                           << "the SIMT buffer view of a " << desc->elementType << " tensor starts at element "
                           << *constant << ", which is not a whole number of bytes; packed " << logicalBits
                           << "-bit elements share one byte, so the view start must be a multiple of "
                           << storageElementsPerLogicalElement;
            }
            elementsPerStorageUnit.push_back(storageElementsPerLogicalElement);
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
            } else if (viewOffset) {
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
            // The parameter must carry an *integer* address space, not
            // #hivm.address_space<...>: only then does memref-to-llvm lower it to a
            // bare pointer matching the !llvm.ptr the launch passes. With the hivm
            // attribute it becomes a descriptor struct and hivmc-a5 aborts with
            // "Calling a function with a bad signature!".
            paramTypes.push_back(MemRefType::get(
                paramShape, baseType.getElementType(), MemRefLayoutAttrInterface{},
                rewriter.getI64IntegerAttr(*addressSpace)));
        }

        // Raw-pointer captures: one parameter per memref a t.ptr chain reads,
        // statically shaped and carrying the integer address space the launch
        // ABI lowers to a bare pointer. No view offset folds in -- t.ptr is the
        // raw storage base -- so the launch passes the un-advanced pointer.
        //
        // The parameter is given an identity layout on purpose: the outlined
        // body only extracts its address (extract_aligned_pointer_as_index),
        // never an index, so a strided source memref is fine here as long as
        // its shape is static. A memref the region actually indexes is rejected
        // by checkSimtRegionCaptures instead of being flattened this way.
        SmallVector<std::pair<Value, int>, 4> pointerParams;
        for (Value buffer : captures.pointerBuffers) {
            auto bufferType = cast<MemRefType>(buffer.getType());
            std::optional<int> addressSpace = simtAddressSpaceForMemref(bufferType);
            if (!addressSpace)
                return failure(); // reported by checkSimtRegionCaptures before the driver ran
            paramTypes.push_back(MemRefType::get(
                bufferType.getShape(), bufferType.getElementType(), MemRefLayoutAttrInterface{},
                rewriter.getI64IntegerAttr(*addressSpace)));
            pointerParams.push_back({buffer, *addressSpace});
        }

        // ---- the outlined vector function ----
        auto parentFunc = vecFuncOp->getParentOfType<func::FuncOp>();
        std::string vfName = (parentFunc ? parentFunc.getName().str() : std::string("kernel")) + "_vf_simt" +
                             (nextSimtRegionId ? std::to_string(nextSimtRegionId) : std::string());
        ++nextSimtRegionId;

        for (Value scalar : captures.scalars) {
            Type type = scalar.getType();
            if (auto integer = dyn_cast<IntegerType>(type))
                type = rewriter.getIntegerType(integer.getWidth());
            paramTypes.push_back(type);
        }

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
        for (auto [buffer, arg] : llvm::zip_equal(
                 captures.pointerBuffers, vfArgs.slice(captures.tensors.size(), captures.pointerBuffers.size())))
            mapping.map(buffer, arg);
        for (auto [scalar, arg] : llvm::zip_equal(
                 captures.scalars,
                 vfArgs.slice(captures.tensors.size() + captures.pointerBuffers.size(), captures.scalars.size()))) {
            Value mapped = arg;
            if (arg.getType() != scalar.getType())
                mapped = bodyBuilder.create<UnrealizedConversionCastOp>(loc, scalar.getType(), arg).getResult(0);
            mapping.map(scalar, mapped);
        }
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
        if (failed(rewriteTensorPtrWithCoordOntoParams(vf)))
            return failure();
        lowerSimtTripleOpsIn<
            ::tla::ThreadIdxOp, hivm_regbaseintrins::ThreadIdXOp, hivm_regbaseintrins::ThreadIdYOp,
            hivm_regbaseintrins::ThreadIdZOp>(vf);
        lowerSimtTripleOpsIn<
            ::tla::ThreadBlockDimOp, hivm_regbaseintrins::BlockDimXOp, hivm_regbaseintrins::BlockDimYOp,
            hivm_regbaseintrins::BlockDimZOp>(vf);
        lowerSimtArithmeticIn(vf);
        if (failed(lowerSimtWarpOpsIn(vf, blockDimX, blockDimY, blockDimZ)))
            return failure();
        lowerSimtSyncThreadsIn(vf);
        lowerSimtShuffleIn(vf);

        // ---- the launch ----
        SmallVector<Value, 4> launchArgs;
        for (auto [base, addressSpace, viewOffset, elementType, elementsPerUnit] :
             llvm::zip_equal(baseMemrefs, addressSpaces, viewOffsets, elementTypes, elementsPerStorageUnit)) {
            Value pointer = memrefToRawPointer(callBuilder, loc, base, addressSpace);
            if (viewOffset) {
                Value index = callBuilder.createOrFold<arith::IndexCastOp>(loc, callBuilder.getI64Type(), viewOffset);
                if (elementsPerUnit > 1) {
                    // The GEP steps in storage elements (bytes for packed fp4),
                    // so convert the logical element offset first.
                    Value divisor = callBuilder.create<arith::ConstantIntOp>(loc, elementsPerUnit, 64);
                    index = callBuilder.createOrFold<arith::DivSIOp>(loc, index, divisor);
                }
                auto pointerType = LLVM::LLVMPointerType::get(callBuilder.getContext(), addressSpace);
                pointer = callBuilder.create<LLVM::GEPOp>(loc, pointerType, elementType, pointer, ValueRange{index});
            }
            launchArgs.push_back(pointer);
        }
        // Raw-pointer launch arguments: the un-advanced storage base, with no
        // view GEP -- t.ptr must stay the raw base on the other side.
        for (auto [buffer, addressSpace] : pointerParams)
            launchArgs.push_back(memrefToRawPointer(callBuilder, loc, buffer, addressSpace));
        // Captured runtime scalars follow the buffers and raw pointers, in
        // capture order: the ABI is positional, so this must match the
        // parameter list built above.
        for (Value scalar : captures.scalars) {
            if (auto integer = dyn_cast<IntegerType>(scalar.getType()); integer && !integer.isSignless())
                scalar =
                    callBuilder
                        .create<UnrealizedConversionCastOp>(loc, callBuilder.getIntegerType(integer.getWidth()), scalar)
                        .getResult(0);
            launchArgs.push_back(scalar);
        }
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
            if (isa<::tla::SimtLoadOp, ::tla::SimtStoreOp, ::tla::SimtAtomicAddOp>(op))
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
            } else if (auto atomicAdd = dyn_cast<::tla::SimtAtomicAddOp>(op)) {
                if (failed(lowerSimtAtomicAdd(builder, atomicAdd)))
                    return failure();
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

        // The pointer chain the atomics consumed has no reader left: the address
        // was rebuilt from the memref parameter or the captured integer. Reverse
        // walk order drops the outer ptr_add before the root it still reads.
        SmallVector<Operation*, 8> chain;
        vf.walk([&](Operation* op) {
            if (isa<::tla::PtrAddOp, ::tla::TensorPtrOp, ::tla::IntToPtrOp, ::tla::RecastPtrOp>(op))
                chain.push_back(op);
        });
        for (auto it = chain.rbegin(); it != chain.rend(); ++it)
            if ((*it)->use_empty())
                (*it)->erase();
        return success();
    }

    // Fold tla.tensor_ptr_with_coord onto the buffer parameter its source tensor
    // was mapped to: the launch pointer already carries the view start, so the
    // parameter's aligned pointer is exactly raw + coord*stride. Consumers hold
    // i64 addresses at this point -- lower-ptr converted every !tla.ptr use --
    // so the chain rebuilds the typed pointer and the ptrtoint(IntToPtr(i64))
    // pair folds straight back to the address.
    static LogicalResult rewriteTensorPtrWithCoordOntoParams(func::FuncOp vf)
    {
        SmallVector<::tla::TensorPtrWithCoordOp, 8> stale;
        vf.walk([&](::tla::TensorPtrWithCoordOp op) { stale.push_back(op); });
        for (::tla::TensorPtrWithCoordOp op : stale) {
            OpBuilder builder(op);
            // The clone remapped the tensor operand to a memref parameter; read
            // it positionally like rewriteSimtAccessesOntoParams does.
            Value source = op->getOperand(0);
            if (!isa<MemRefType>(source.getType()))
                return op.emitOpError("tla.tensor_ptr_with_coord source did not map to a memref parameter"), failure();
            Value addressIndex = builder.create<mlir::memref::ExtractAlignedPointerAsIndexOp>(op.getLoc(), source);
            Value addressI64 =
                builder.create<mlir::arith::IndexCastOp>(op.getLoc(), builder.getI64Type(), addressIndex);
            Value pointer = builder.create<::tla::IntToPtrOp>(op.getLoc(), op.getResult().getType(), addressI64);
            op.getResult().replaceAllUsesWith(pointer);
            op.erase();
        }

        // tla.ptrtoint(tla.inttoptr(i64)) is the address itself.
        foldIntToPtrPtrToIntRoundTrips(vf.getOperation());
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
        // Capture indices are one-based and shared by buffer and raw-pointer
        // captures, so "#N of M" names one capture unambiguously. Raw-pointer
        // captures are marked as such and carry their address space like the
        // tensor branch does.
        SmallVector<std::string, 4> nonGm;
        SmallVector<std::string, 4> dynamic;
        unsigned captureIndex = 0;
        for (Value tensor : captures.tensors) {
            ++captureIndex;
            auto descOp = tensor.getDefiningOp<::tla::TensorDescOp>();
            if (!descOp)
                continue;
            FailureOr<TensorDescriptor> desc = descriptorFromTensorDescOp(descOp);
            if (failed(desc))
                continue;
            if (!simtAddressSpaceFor(desc->addrspace))
                nonGm.push_back((Twine("#") + Twine(captureIndex) + " in '" + desc->addrspace + "'").str());
            else if (!getConstantIntValue(desc->shape[1]))
                dynamic.push_back((Twine("#") + Twine(captureIndex)).str());
        }
        // Raw-pointer captures carry the same two constraints as buffer
        // captures: a reachable address space and a static shape.
        for (Value buffer : captures.pointerBuffers) {
            ++captureIndex;
            auto bufferType = cast<MemRefType>(buffer.getType());
            std::string label = (Twine("#") + Twine(captureIndex) + " (raw pointer)").str();
            if (!simtAddressSpaceForMemref(bufferType))
                nonGm.push_back((Twine(label) + " in '" + memrefAddressSpaceName(bufferType) + "'").str());
            else if (llvm::any_of(bufferType.getShape(), ShapedType::isDynamic))
                dynamic.push_back(label);
        }
        // A memref the region indexes directly cannot cross the launch ABI
        // faithfully. The call passes one bare pointer per buffer and no
        // layout, so the outlined body would re-interpret every index under an
        // identity layout and silently address the wrong element of a strided
        // or dynamic-stride buffer. Only the t.ptr address read is layout-free.
        if (!captures.directMemrefs.empty()) {
            std::string bufferList;
            llvm::raw_string_ostream rawBufferList(bufferList);
            llvm::interleaveComma(captures.directMemrefs, rawBufferList, [&](Value buffer) {
                buffer.printAsOperand(rawBufferList, OpPrintingFlags());
                rawBufferList << " : ";
                buffer.getType().print(rawBufferList);
            });
            vecFuncOp.emitOpError()
                << "a SIMT tla.vec.func can only read a buffer through its tensor: this region indexes "
                << captures.directMemrefs.size() << " raw memref(s) directly (" << llvm::StringRef(rawBufferList.str())
                << "). The launch ABI passes one bare pointer per buffer and no layout, so the outlined "
                   "body would re-interpret every index under an identity layout and can silently read "
                   "the wrong element of a strided or dynamic-stride buffer. Index the tensor instead "
                   "(tensor[i] / tensor[i] = value), or read the storage address through t.ptr";
            result = failure();
        }

        // A SIMT vector function is launched with GM pointers only. A UB operand
        // never gets a base established and silently reads as zeros, so refuse it
        // rather than let the kernel produce wrong answers.
        if (!nonGm.empty()) {
            vecFuncOp.emitOpError() << "a SIMT tla.vec.func can only address GM or UB, but " << nonGm.size() << " of "
                                    << (captures.tensors.size() + captures.pointerBuffers.size())
                                    << " capture(s) used by this region are not: " << llvm::join(nonGm, ", ")
                                    << ". Buffers reach an outlined SIMT vector function as raw pointers, and "
                                       "only the GM and UB address spaces have a SIMT pointer form";
            result = failure();
        }
        // One pointer per buffer crosses the launch ABI, so the parameter must be a
        // statically shaped memref: a dynamic extent lowers to a 5-field descriptor
        // and hivmc-a5 aborts with an opaque LLVM assertion.
        if (!dynamic.empty()) {
            vecFuncOp.emitOpError() << "a SIMT tla.vec.func requires statically shaped buffers, but capture(s) "
                                    << llvm::join(dynamic, ", ") << " of "
                                    << (captures.tensors.size() + captures.pointerBuffers.size())
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
