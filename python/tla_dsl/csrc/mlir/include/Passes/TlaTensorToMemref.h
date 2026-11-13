#pragma once

// Consumer-side materialization of already-lowered tensor descriptors.
//
// Raw tensor producer analysis belongs to TlaTensorDescDerivation and is not
// exposed through this interface.

#include "Passes/TlaTensorDescriptor.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

#include <string>

namespace tla {

/// Widen/narrow an integer/index value to i64 (used for runtime-call payloads).
mlir::Value castValueToI64(mlir::OpBuilder& builder, mlir::Location loc, mlir::Value value);

/// Cast a memref value to `memrefType` via `memref.cast`; identity if the type
/// already matches, failure if `value` is not a memref.
mlir::FailureOr<mlir::Value> castMemrefToType(
    mlir::OpBuilder& builder, mlir::Location loc, mlir::Value value, mlir::MemRefType memrefType);

// ---------------------------------------------------------------------------
// Copy-route runtime lowering (shared by tla-cube-region and the tla.vector
// lowering passes).
// A tla.copy between two on-/off-chip layouts lowers to an inlinable C-interface
// runtime template (bc/Cube or bc/Vector). These helpers pick the template name,
// build its shape/stride/coord payload from the operand descriptors, and declare
// (and core-annotate) the template symbol. The cube pass uses them for the
// L1/L0A/L0B/L0C routes; the vector pass uses them for the UB->L1 route.
// ---------------------------------------------------------------------------

/// Runtime template symbol for a copy between (src,dst) addrspace + layout tags,
/// or "" if the (route, element-type) combination is unsupported. `extraDesc` is
/// the L0C->UB split-mode infix.
std::string getCopyRouteCallee(
    mlir::MLIRContext* ctx, llvm::StringRef srcAddrspace, llvm::StringRef dstAddrspace, ::LayoutTag srcLayout,
    ::LayoutTag dstLayout, mlir::Type srcElementType, mlir::Type dstElementType, llvm::StringRef extraDesc = "");

/// Runtime template symbol for an L1 fill of an element type, or "" if the
/// (layout-tag, element-type) combination is unsupported. Symbols are
/// `fill_l1_<zN|nZ>_<elem-suffix>`.
std::string getFillCallee(::LayoutTag dstLayout, mlir::Type dstElementType);

/// The 24-element i64 runtime payload for a copy route: the src tile's 12-field
/// (4D) layout descriptor followed by the dst tile's. Linear descriptors carry
/// shape[2]=shape[3]=stride[2]=stride[3]=1 (enforced by validateTensorDescriptor),
/// so both Linear and NZFamily endpoints use the same 12-field encoding.
llvm::SmallVector<mlir::Value, 24> buildCopyPayloadForRoute(
    mlir::OpBuilder& builder, mlir::Location loc, const TensorDescriptor& srcDesc, const TensorDescriptor& dstDesc);

/// Append the trailing ``i64 gmCachePolicy`` for a GM-side BC copy stub.
/// Reads ``src_cache_policy`` / ``dst_cache_policy`` from ``copyOp`` (default NORMAL=1).
/// No-op when neither endpoint is GM.
void appendGmCachePolicyOperand(
    mlir::OpBuilder& builder, mlir::Location loc, ::tla::CopyOp copyOp, llvm::StringRef srcAddrspace,
    llvm::StringRef dstAddrspace, llvm::SmallVectorImpl<mlir::Type>& operandTypes,
    llvm::SmallVectorImpl<mlir::Value>& operands);

/// Declare (once) the private runtime template `name`, core-annotating it (AIC or
/// AIV, always-inline, C-interface) by symbol so the backend inlines it on the
/// right core. Returns the existing decl if already present.
mlir::func::FuncOp getOrCreateRuntimeCall(
    mlir::ModuleOp module, llvm::StringRef name, llvm::ArrayRef<mlir::Type> operandTypes,
    llvm::ArrayRef<mlir::Type> resultTypes = {});

/// Per-consumer state for materializing memrefs from descriptor values.
class TlaTensorMemrefLowering {
public:
    llvm::DenseMap<mlir::Value, TensorDescriptor> descriptorByValue;
    llvm::DenseMap<mlir::Value, mlir::Value> loweredMemrefByValue;
};

// ---------------------------------------------------------------------------
// Base-memref materialization: turn a descriptor base (a memref or an
// address-backed `!tla.ptr`) into a concrete memref of the wanted type.
// ---------------------------------------------------------------------------

void pushStagedErase(llvm::SmallVectorImpl<mlir::Operation*>* toErase, mlir::Operation* op);
void pushStagedErase(llvm::SmallVectorImpl<mlir::Operation*>& toErase, mlir::Operation* op);

/// True iff every result user of `op` is already staged for erasure.
bool hasOnlyStagedResultUsers(mlir::Operation* op, llvm::ArrayRef<mlir::Operation*> stagedErase);
/// Transitively stage materialized tensor descriptors whose only users are
/// staged for erasure.
void stageDeadTensorDescriptors(mlir::Operation* root, llvm::SmallVectorImpl<mlir::Operation*>& toErase);

/// Materialize the `tla.inttoptr` boundary produced by `tla-lower-ptr` as
/// `memrefType`. `dynamicSizes` describes dynamic dimensions of `memrefType`.
mlir::FailureOr<mlir::Value> materializePtrValueAsMemref(
    mlir::OpBuilder& builder, mlir::Location loc, mlir::Value ptrValue, mlir::MemRefType memrefType,
    mlir::Operation* diagnosticOp, mlir::ValueRange dynamicSizes = {});

/// Materialize `desc.base` as a concrete memref of `desc.bridgedBaseMemrefType`,
/// using the pointer address in the descriptor. Dynamic GM dimensions use
/// descriptor shape SSA values; every dynamic dimension of a ptr-backed
/// on-chip base uses the same zero value as an unknown-capacity ABI sentinel,
/// without imposing a rank restriction.
mlir::FailureOr<mlir::Value> materializeDescriptorBaseMemref(
    mlir::OpBuilder& builder, mlir::Location loc, const TensorDescriptor& desc, mlir::Operation* diagnosticOp);

/// Lower-once wrapper over `materializeDescriptorBaseMemref`. On the first call
/// for a given `desc.base`, the base memref is materialized at a point that
/// dominates every use of `desc.base` (right after its defining op, or the entry
/// of its block for a block argument) and cached in `baseMemrefCache`. Later
/// calls with the same base return the cached memref (cast to the wanted type at
/// the caller). Because the cached value dominates all uses of `desc.base`, it
/// dominates every consumer, so the reuse is SSA-safe.
mlir::FailureOr<mlir::Value> getOrMaterializeDescriptorBaseMemref(
    mlir::OpBuilder& builder, mlir::Location loc, const TensorDescriptor& desc, mlir::Operation* diagnosticOp,
    llvm::DenseMap<mlir::Value, mlir::Value>& baseMemrefCache);

/// Collect descriptors already materialized as `tla.tensor_desc` in `funcOp`.
/// This is the read-only consumer entry point for Cube/Vector lowering: it never
/// derives metadata from a raw tile producer and rejects any such producer that
/// crosses the `tla-lower-tensor-desc` pass boundary.
mlir::LogicalResult collectMaterializedTensorDescriptors(
    mlir::func::FuncOp funcOp, llvm::DenseMap<mlir::Value, TensorDescriptor>& descriptorByValue);

/// Materialize a tile's memref directly from its `TensorDescriptor`: the base
/// memref (via `getOrMaterializeDescriptorBaseMemref`) for a rank-1 flattened
/// buffer, otherwise a `memref.subview` at the tile's (row,col) offset and shape.
/// Descriptor-driven tile materialization shared by tile subview lowering and
/// the tla.mmad operand path. The copy routes (tla.copy / tla.copy_mx) call
/// `getOrMaterializeDescriptorBaseMemref` themselves instead: they ship the tile
/// coordinate in the runtime payload, and the bc layer rebases on it.
///
/// `foldRankOneCoord` decides who owns a rank-1 tile's coordinate. The copy
/// routes ship it in the descriptor payload and let the bc layer compute the
/// address, so folding it into the memref as well would offset twice; tla.mmad
/// hands the memref straight to a wrapper that never looks at a coordinate, so
/// it is the one caller that has to have the offset baked in.
mlir::FailureOr<mlir::Value> materializeTileMemrefFromDescriptor(
    mlir::OpBuilder& builder, mlir::Location loc, const TensorDescriptor& desc, mlir::Operation* diagnosticOp,
    llvm::DenseMap<mlir::Value, mlir::Value>& baseMemrefCache, bool foldRankOneCoord = false);

// ---------------------------------------------------------------------------
// Vector tile memref materialization (the tla.vector lowering passes). Vector-specific
// policy — 256-byte lane memrefs, rank-1<->2 reshape, the per-copy handoff cache
// — layered on the shared decode/bridge above.
// ---------------------------------------------------------------------------

mlir::FailureOr<int64_t> getStaticNumElements(llvm::ArrayRef<int64_t> shape);
llvm::SmallVector<mlir::Value, 12> buildCopyPayloadForDescriptor(
    mlir::OpBuilder& builder, mlir::Location loc, const TensorDescriptor& desc);

mlir::FailureOr<int64_t> getElementByteWidth(mlir::Type elementType);
mlir::FailureOr<int64_t> getVectorLaneCount(mlir::Type elementType);

/// Bridge `tensor`'s `!tla.tensor` type to a memref (Value form of the bridge).
mlir::FailureOr<mlir::MemRefType> getBridgedTensorMemrefType(mlir::Value tensor);

/// Preferred memref type for a vector-helper operand (flat make_tensor_like or bridged).
mlir::FailureOr<mlir::MemRefType> getVectorHelperArgMemrefType(mlir::Value operand);

/// Adapt a memref to `expectedType`: rank-1<->2 reshape when the element storage
/// matches, else a plain memref.cast.
mlir::FailureOr<mlir::Value> castMemrefToExpected(
    mlir::PatternRewriter& rewriter, mlir::Location loc, mlir::Value value, mlir::MemRefType expectedType);

/// Materialize the base memref a tensor/tile views into (handoff cache, ptr
/// materialization and tile-view recursion).
mlir::FailureOr<mlir::Value> materializeBaseMemref(
    mlir::PatternRewriter& rewriter, mlir::Location loc, mlir::Value tensor,
    llvm::DenseMap<mlir::Value, mlir::Value>* loweredMemrefByValue = nullptr);

/// Build a rank-1, `numElements`-wide reinterpret_cast of `baseMemref` at element
/// `offset` (dynamic stride-1 layout). Used by the vector helper's per-lane tiles.
mlir::Value materializeFlatReinterpretSubview(
    mlir::OpBuilder& builder, mlir::Location loc, mlir::Value baseMemref, mlir::Value offset, int64_t numElements);

} // namespace tla
