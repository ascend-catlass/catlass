#pragma once

#include <optional>

#include "Dialect/Tla/IR/TlaAttrs.h"
#include "Dialect/Tla/IR/TlaDialect.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Types.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"

#define GET_TYPEDEF_CLASSES
#include "tla/Types.h.inc"

namespace tla {
::mlir::LogicalResult getTlaIndexTreeLeaves(::llvm::ArrayRef<int64_t> tree, ::llvm::SmallVectorImpl<int64_t>& leaves);

/// GM-side MX scale blocks. NZFamily descriptors -- the 2x2 nested tree of
/// tla::MakeMxScaleLayout with the plain matrix dimension spelled as a
/// (1, dim) pair whose leading leaf carries stride 0. The tag records which
/// side (A / B) and orientation, which is what selects the copy; the bc layer
/// re-derives the rank structure Catlass's copy predicates key on device-side.
inline bool isMxScaleGmLayout(::LayoutTag layoutTag)
{
    return layoutTag == ::LayoutTag::RowMajorMxScaleA || layoutTag == ::LayoutTag::ColMajorMxScaleA ||
           layoutTag == ::LayoutTag::RowMajorMxScaleB || layoutTag == ::LayoutTag::ColMajorMxScaleB;
}

/// The single source of truth for layouts whose shape and stride each have
/// four physical leaves in tensor descriptors.
inline bool isNZFamilyLayout(::LayoutTag layoutTag)
{
    return layoutTag == ::LayoutTag::zN || layoutTag == ::LayoutTag::nZ || layoutTag == ::LayoutTag::L0Clayout ||
           layoutTag == ::LayoutTag::zNUnAlign || layoutTag == ::LayoutTag::zZMxScale ||
           layoutTag == ::LayoutTag::nNMxScale || isMxScaleGmLayout(layoutTag);
}

/// A flat, unit-stride run of elements: rank-1 shape/stride, a single coordinate.
///
/// This is the tag that lines up with `tla::detail::isVector` on the tile side
/// (depth 1, rank 1, stride 0 == 1), so the VectorLayout TileCopyTla
/// specializations serve a VectorLayout tile without the bc layer having to project a
/// two-dimensional layout down to one. RowMajor deliberately stays rank-2: a
/// flat tile is spelled with this tag rather than as a (1, N) RowMajor one.
inline bool isVectorLayout(::LayoutTag layoutTag)
{
    return layoutTag == ::LayoutTag::VectorLayout;
}

/// Linear layouts carry a flat (non-fractal) shape/stride pair. VectorLayout is one of
/// them, so every linear rule -- rank-1 or rank-2 metadata, unit padding in
/// descriptor slots 2 and 3, `layout()(coord)` addressing -- covers it too.
///
/// This is the single classifier for that family; the descriptor/schema code and
/// the dialect verifiers all ask it rather than re-listing the three tags.
inline bool isLinearLayout(::LayoutTag layoutTag)
{
    return layoutTag == ::LayoutTag::RowMajor || layoutTag == ::LayoutTag::ColumnMajor || isVectorLayout(layoutTag);
}

inline int64_t getByteSizeOfFixedWidthScalarType(::mlir::Type type)
{
    if (type.isBF16() || type.isF16())
        return 2;
    if (type.isF32())
        return 4;
    if (type.isF64())
        return 8;
    // fp8 cube operand formats (f8E4M3FN / f8E5M2) are byte-sized.
    if (::llvm::isa<::mlir::Float8E4M3FNType, ::mlir::Float8E5M2Type>(type))
        return 1;
    // An e8m0 scale block is one opaque byte per shared exponent.
    if (::llvm::isa<::tla::Float8E8M0Type>(type))
        return 1;
    if (auto intTy = ::llvm::dyn_cast<::mlir::IntegerType>(type)) {
        if (intTy.getWidth() % 8 == 0)
            return intTy.getWidth() / 8;
    }
    return 0;
}

/// Width of a scalar element in *bits*.
///
/// The primitive width. getByteSizeOfFixedWidthScalarType above cannot express a
/// sub-byte element and answers 0 for one, which is the right answer for every
/// byte-denominated caller (pointer arithmetic, vector lanes) -- those genuinely
/// cannot address half a byte and should refuse. Callers that convert between an
/// element count and a byte size need this one instead.
inline int64_t getBitSizeOfFixedWidthScalarType(::mlir::Type type)
{
    if (auto intTy = ::llvm::dyn_cast<::mlir::IntegerType>(type))
        return intTy.getWidth();
    // The packed fp4 formats are the sub-byte case getByteSize... cannot express.
    if (::llvm::isa<::tla::Float4E2M1Type, ::tla::Float4E1M2Type>(type))
        return 4;
    return getByteSizeOfFixedWidthScalarType(type) * 8;
}

/// Predicate geometry for a packed vector store.  The factor converts logical
/// destination elements into predicate lanes for the source register.
struct PackedStorePredicateGeometry {
    int64_t sourcePredicateLanesPerDestElement;
};

/// Return the supported non-NORM store predicate geometry for ``storeDist``.
///
/// PACK_B32 always writes one 16-bit payload per B32 slot.  A 16-bit source
/// register contains two source predicate lanes per B32 slot, while a 32-bit
/// source register contains one.  PACK_B16 writes one 8-bit payload per B16
/// slot.  ONEPT distributions preserve one source predicate lane per
/// same-width destination element. The DSL deliberately rejects other
/// combinations rather than inventing a source-to-destination predicate
/// mapping the hardware does not define.
inline std::optional<PackedStorePredicateGeometry> getPackedStorePredicateGeometry(
    ::StoreDist storeDist, ::mlir::Type sourceElementType, ::mlir::Type destElementType)
{
    if (storeDist == ::StoreDist::norm)
        return PackedStorePredicateGeometry{1};

    int64_t sourceBits = getBitSizeOfFixedWidthScalarType(sourceElementType);
    int64_t destBits = getBitSizeOfFixedWidthScalarType(destElementType);
    if (storeDist == ::StoreDist::pack_b32 && destBits == 16) {
        if (sourceBits == 32)
            return PackedStorePredicateGeometry{1};
        if (sourceBits == 16)
            return PackedStorePredicateGeometry{2};
    }
    if (storeDist == ::StoreDist::pack_b16 && sourceBits == 16 && destBits == 8)
        return PackedStorePredicateGeometry{1};
    // One destination byte per 32-bit slot, independent of the source's
    // numeric interpretation. An 8-bit source needs four predicate lanes
    // per output byte; a 16-bit source needs two, and a 32-bit source one.
    if (storeDist == ::StoreDist::pack4_b32 && destBits == 8 &&
        (sourceBits == 8 || sourceBits == 16 || sourceBits == 32))
        return PackedStorePredicateGeometry{32 / sourceBits};
    if ((storeDist == ::StoreDist::first_element_b8 && sourceBits == 8 && destBits == 8) ||
        (storeDist == ::StoreDist::first_element_b16 && sourceBits == 16 && destBits == 16) ||
        (storeDist == ::StoreDist::first_element_b32 && sourceBits == 32 && destBits == 32))
        return PackedStorePredicateGeometry{1};
    return std::nullopt;
}

/// True for a packed 4-bit cube float, either encoding.
inline bool isPackedFp4Type(::mlir::Type type)
{
    return ::llvm::isa<::tla::Float4E2M1Type, ::tla::Float4E1M2Type>(type);
}

/// True for an element format this dialect defines itself because MLIR has no
/// builtin for it. These are buffered as i8 at the memref boundary.
inline bool isTlaCustomElementType(::mlir::Type type)
{
    return isPackedFp4Type(type) || ::llvm::isa<::tla::Float8E8M0Type>(type);
}

using coord = ::mlir::Type;
using cross_flag = ::mlir::Type;
using flag = ::mlir::Type;
using index = ::mlir::IndexType;
using memref = ::mlir::Type;
using mutex = ::mlir::Type;
using range = ::mlir::Type;
using shape = ::mlir::Type;
using stride = ::mlir::Type;
using layout = ::mlir::Type;
using tensor = ::mlir::Type;
using tile = ::mlir::Type;
// PtrType is defined in Types.h.inc (TableGen TypeDef).
} // namespace tla
