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

/// Packed x2 byte carriers, distinct from logical 4-bit FP4 types.
inline bool isFp4PairType(::mlir::Type type)
{
    return ::llvm::isa<::tla::Float4E2M1x2Type, ::tla::Float4E1M2x2Type>(type);
}

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

/// GM-to-L1 layout conversions with a registered batched BC wrapper.
inline bool isSupportedBatchCopyLayout(::LayoutTag srcLayout, ::LayoutTag dstLayout, ::mlir::Type elementType)
{
    if (elementType.isFloat8E4M3FN() || elementType.isFloat8E5M2())
        return (srcLayout == ::LayoutTag::RowMajor && dstLayout == ::LayoutTag::zN) ||
               (srcLayout == ::LayoutTag::ColumnMajor && dstLayout == ::LayoutTag::nZ);
    if (::llvm::isa<::tla::Float8E8M0Type>(elementType))
        return (dstLayout == ::LayoutTag::zZMxScale &&
                (srcLayout == ::LayoutTag::RowMajorMxScaleA || srcLayout == ::LayoutTag::ColMajorMxScaleA)) ||
               (dstLayout == ::LayoutTag::nNMxScale &&
                (srcLayout == ::LayoutTag::RowMajorMxScaleB || srcLayout == ::LayoutTag::ColMajorMxScaleB));
    return false;
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

inline bool isOfficialL1FillElementType(::mlir::Type type)
{
    return type.isF16() || type.isBF16() || type.isF32() || type.isSignlessInteger(16) || type.isUnsignedInteger(16) ||
           type.isSignlessInteger(32) || type.isUnsignedInteger(32);
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
    // HiF8, packed FP4 pairs and E8M0 shared exponents occupy one byte.
    if (::llvm::isa<::tla::Float8E8M0Type, ::tla::HiFloat8Type, ::tla::Float4E2M1x2Type, ::tla::Float4E1M2x2Type>(type))
        return 1;
    if (auto intTy = ::llvm::dyn_cast<::mlir::IntegerType>(type)) {
        if (intTy.getWidth() % 8 == 0)
            return intTy.getWidth() / 8;
    }
    return 0;
}

/// True for IR Bool (`i1`). Logical identity only — physical storage is b8
/// (`elementStorageBytes` → 1); AscendC `TypeGet<bool>` → `vector_s8`.
inline bool isBoolElementType(::mlir::Type type)
{
    if (auto intTy = ::llvm::dyn_cast<::mlir::IntegerType>(type))
        return intTy.getWidth() == 1;
    return false;
}

/// Byte width used for VectorSSA / MaskSSA lane capacity, pointer scaling, and
/// allocation element counts (VL = 256B / bytes).
///
/// AscendC `TypeGet<bool>` → `vector_s8` and `CreateMask<bool>` take the
/// `sizeof(T)==1` / `pset_b8` path, so Bool occupies **1 byte per lane** even
/// though IR keeps an `i1` element identity. `getByteSizeOfFixedWidthScalarType`
/// returns 0 for i1 (not byte-aligned); this helper maps i1 → 1 and otherwise
/// forwards to that helper. **Prefer this over `getBitSize…` for any
/// byte-denominated math.**
inline int64_t elementStorageBytes(::mlir::Type type)
{
    if (isBoolElementType(type))
        return 1;
    return getByteSizeOfFixedWidthScalarType(type);
}

/// Storage width in bits (`elementStorageBytes * 8`). Use for stride alignment
/// quanta; do not confuse with `getBitSizeOfFixedWidthScalarType` (logical bits,
/// where Bool is 1).
inline int64_t elementStorageBits(::mlir::Type type)
{
    int64_t bytes = elementStorageBytes(type);
    return bytes > 0 ? bytes * 8 : 0;
}

/// Logical width of a scalar element in *bits* (Bool → 1, packed fp4 → 4).
///
/// Not for capacity / pointer arithmetic — those use `elementStorageBytes` /
/// `elementStorageBits`. `getByteSizeOfFixedWidthScalarType` answers 0 for
/// sub-byte types; callers that convert element counts to byte sizes of true
/// packed formats need this one instead.
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
/// PACK_B16/B32/B64 write the lower half of each hardware lane. For a
/// same-type tile, each destination element has room for two source lanes.
/// PACK4_B32 writes one byte per 32-bit slot, including views that already
/// use a narrower element type for those slots.
inline std::optional<PackedStorePredicateGeometry> getPackedStorePredicateGeometry(
    ::StoreDist storeDist, ::mlir::Type sourceElementType, ::mlir::Type destElementType)
{
    if (storeDist == ::StoreDist::norm)
        return PackedStorePredicateGeometry{1};

    // Storage widths (Bool i1 → 8 bits). Logical getBitSize(Bool)==1 is wrong
    // for Dist layout checks that key off b8/b16/b32/b64 movers.
    int64_t sourceBits = elementStorageBits(sourceElementType);
    int64_t destBits = elementStorageBits(destElementType);
    if (sourceElementType == destElementType) {
        if ((storeDist == ::StoreDist::pack_b16 && (sourceBits == 8 || sourceBits == 16)) ||
            (storeDist == ::StoreDist::pack_b32 && (sourceBits == 16 || sourceBits == 32)) ||
            (storeDist == ::StoreDist::pack_b64 && sourceBits == 64))
            return PackedStorePredicateGeometry{2};
        if (storeDist == ::StoreDist::pack4_b32 && sourceBits == 32)
            return PackedStorePredicateGeometry{4};
    }

    if (storeDist == ::StoreDist::pack4_b32 && destBits == 8 &&
        (sourceBits == 8 || sourceBits == 16 || sourceBits == 32))
        return PackedStorePredicateGeometry{32 / sourceBits};
    if ((storeDist == ::StoreDist::first_element_b8 && sourceBits == 8 && destBits == 8) ||
        (storeDist == ::StoreDist::first_element_b16 && sourceBits == 16 && destBits == 16) ||
        (storeDist == ::StoreDist::first_element_b32 && sourceBits == 32 && destBits == 32))
        return PackedStorePredicateGeometry{1};
    if ((storeDist == ::StoreDist::intlv_b8 || storeDist == ::StoreDist::intlv_b16 ||
         storeDist == ::StoreDist::intlv_b32) &&
        sourceBits == destBits && sourceBits > 0)
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
    return isPackedFp4Type(type) ||
           ::llvm::isa<::tla::Float8E8M0Type, ::tla::HiFloat8Type, ::tla::Float4E2M1x2Type, ::tla::Float4E1M2x2Type>(
               type);
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
