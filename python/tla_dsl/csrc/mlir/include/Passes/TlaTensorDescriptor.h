#pragma once

// Shared representation and codec for materialized tensor descriptors.
//
// `tla-lower-tensor-desc` owns producer-chain analysis. Downstream passes may
// decode and validate `tla.tensor_desc`, but must not recover descriptor
// metadata by walking raw tensor producers.

#include "Dialect/Tla/IR/TlaOps.h"
#include "Dialect/Tla/IR/TlaTypes.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

#include <array>
#include <cstdint>
#include <string>

namespace tla {

/// Discardable metadata carried by lowered alloc addresses until tensor views
/// have recovered the static allocation capacity. Removed by finalize.
inline constexpr llvm::StringLiteral kAllocSizeBytesMetadataAttrName = "tla.alloc_size_bytes";

/// Bridge a structured `!tla.tensor` type to a builtin memref type.
mlir::FailureOr<mlir::MemRefType> bridgeTlaTensorType(mlir::Type tlaTensorType);

/// Static metadata decoded from a structured `!tla.tensor` type.
struct TensorTypeInfo {
    llvm::SmallVector<int64_t, 4> shapeDims;
    llvm::SmallVector<int64_t, 4> strideDims;
    llvm::SmallVector<int64_t, 4> coordDims;
    llvm::SmallVector<int64_t, 4> originShapeDims;
    std::string addressSpace;
    mlir::Type elementType;
    ::AddressSpace tlaAddressSpace = ::AddressSpace::gm;
    ::LayoutTag layoutTag = ::LayoutTag::Unknown;
    int64_t rank = 0;
    bool hasBatch = false;
    std::array<int64_t, 4> batchDims{}; // shape, stride, origin, coordinate
};

mlir::FailureOr<TensorTypeInfo> decodeTensorTypeInfo(mlir::Type tensorType);

/// Raw, rank-preserving decode used by vector lowering.
struct ParsedTensorInfo {
    llvm::SmallVector<int64_t, 2> shape;
    llvm::SmallVector<int64_t, 2> originShape;
    llvm::SmallVector<int64_t, 2> coord;
    llvm::SmallVector<int64_t, 2> strides;
    ::AddressSpace addressSpace;
    mlir::Type elementType;
    ::LayoutTag layoutTag = ::LayoutTag::Unknown;
};

mlir::FailureOr<ParsedTensorInfo> parseTensorInfo(mlir::Type tensorType);

/// SSA-valued descriptor represented by `tla.tensor_desc`.
struct TensorDescriptor {
    mlir::Value base;
    mlir::Type bridgedBaseMemrefType;
    std::array<mlir::Value, 4> shape;
    std::array<mlir::Value, 4> stride;
    std::array<mlir::Value, 2> originShape;
    std::array<mlir::Value, 2> coord;
    ::LayoutTag layoutTag = ::LayoutTag::Unknown;
    std::string addrspace;
    mlir::Type elementType;
    llvm::SmallVector<mlir::Value, 4> batch; // optional leading shape/stride/origin/coordinate
};

/// Fully dynamic shape/stride form used at structural joins and runtime calls.
mlir::MemRefType getDynamicStridedMemrefType(mlir::MemRefType memrefType);

bool validateTensorDescriptor(mlir::Operation* op, const TensorDescriptor& desc, llvm::StringRef errorMessage);

/// Recover optional allocation capacity preserved by pointer lowering.
mlir::FailureOr<int64_t> getStaticAllocationElementCount(mlir::Value ptr);

/// Decode an already materialized `tla.tensor_desc`.
mlir::FailureOr<TensorDescriptor> descriptorFromTensorDescOp(::tla::TensorDescOp op);

/// Element offset of a descriptor's view within its allocation, or a null
/// Value when the view starts at the allocation's own base: the formula
/// ``coord[0]*stride[0] + coord[1]*stride[1]`` (elements) that the SIMT launch
/// GEP and ``tla.tensor_ptr_with_coord`` lowering must share, or the same
/// tensor would mean different addresses on the two paths. A descriptor always
/// carries both coordinate and stride leaves, so this cannot fail.
mlir::Value descriptorViewElementOffset(mlir::OpBuilder& builder, mlir::Location loc, const TensorDescriptor& desc);

} // namespace tla
