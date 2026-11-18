#pragma once

#include <type_traits>

#include "common.h"

#include "catlass/arch/arch.hpp"
#include "catlass/layout/layout.hpp"
#include "catlass/numeric_size.hpp"
#include "tla/layout.hpp"
#include "tla/tensor.hpp"

using bf16 = bfloat16_t;

/// Layout tags for the bc-layer DMA helpers. Mirrors the Tla.td definition.
// csrc/mlir/build/tblgen/tla/Enums.h.inc
enum class LayoutTag : uint32_t
{
    Unknown = 0,
    RowMajor = 1,
    ColumnMajor = 2,
    zN = 3,
    nZ = 4,
    L0Clayout = 5,
    zNUnAlign = 6,
    zZMxScale = 7,
    nNMxScale = 8,
    RowMajorMxScaleA = 9,
    ColMajorMxScaleA = 10,
    RowMajorMxScaleB = 11,
    ColMajorMxScaleB = 12,
    VectorLayout = 13,
};

/// Unified 4D (12-field) tensor descriptor. Linear layouts (RowMajor/
/// ColumnMajor) carry shape2=shape3=stride2=stride3=1, so a single descriptor
/// shape serves both Linear and NZFamily endpoints.
struct TensorDesc {
    uint32_t shape0;
    uint32_t shape1;
    uint32_t shape2;
    uint32_t shape3;
    int64_t stride0;
    int64_t stride1;
    int64_t stride2;
    int64_t stride3;
    uint32_t coord0;
    uint32_t coord1;
    uint32_t originShape0;
    uint32_t originShape1;
};

template <typename T, size_t Dim>
CATLASS_DEVICE T* basePtr(memref_t<T, Dim>* memref)
{
    return memref->aligned + memref->offset;
}

template <typename T, size_t Dim>
CATLASS_DEVICE uint32_t localAddr(memref_t<T, Dim>* memref)
{
    return static_cast<uint32_t>(reinterpret_cast<uintptr_t>(basePtr(memref)));
}

template <typename T, size_t Dim>
CATLASS_DEVICE uint32_t elementCount(memref_t<T, Dim>* memref)
{
    // Dynamic ptr-backed on-chip memrefs deliberately carry zero as an unknown
    // allocation-capacity sentinel. LocalTensor consumers use the address plus
    // the separately supplied TensorDesc shape/layout; this count is not part of
    // the copy operation's logical-shape contract.
    uint32_t n = 1;
    for (size_t i = 0; i < Dim; ++i)
        n *= static_cast<uint32_t>(memref->sizes[i]);
    return n;
}

/// CachePolicy integer codes for SetL2CacheHint (DISABLE=0, NORMAL=1).
/// Matches the trailing i64 on GM-side BC copy stubs and Tla CachePolicyAttr.
constexpr int64_t kCachePolicyNormal = 1;

template <typename T>
CATLASS_DEVICE AscendC::GlobalTensor<T> makeGlobalTensor(__gm__ T* ptr, int64_t cachePolicy = kCachePolicyNormal)
{
    AscendC::GlobalTensor<T> tensor;
    tensor.SetGlobalBuffer(ptr);
    // Tag the GM address with the L2 hint (bit60+). NORMAL clears the tag.
    tensor.SetL2CacheHint(static_cast<AscendC::CacheMode>(cachePolicy));
    return tensor;
}

template <typename DescT>
CATLASS_DEVICE auto makeTlaTileCoord(const DescT& desc)
{
    return tla::MakeCoord(desc.coord0, desc.coord1);
}

template <typename T>
CATLASS_DEVICE auto makeRowMajorTlaLayout(const TensorDesc& desc)
{
    return tla::MakeLayout(
        tla::MakeShape(desc.shape0, desc.shape1), tla::MakeStride(desc.stride0, tla::Int<1>{}),
        tla::MakeShape(desc.originShape0, desc.originShape1));
}

CATLASS_DEVICE auto makeColumnMajorTlaLayout(const TensorDesc& desc)
{
    return tla::MakeLayout(
        tla::MakeShape(desc.shape0, desc.shape1), tla::MakeStride(tla::Int<1>{}, desc.stride1),
        tla::MakeShape(desc.originShape0, desc.originShape1));
}

// A row-major tile read as one flat contiguous run. VectorLayout is rank-1, so
// the descriptor's two dimensions collapse into a single index: the extent is
// the row width, and the row coordinate folds into the base offset. layout()
// then yields the same address the rank-2 RowMajor layout would.
//
// The stride is a literal 1, not desc.stride1: `tla::detail::isVector` reads
// `stride<0>()` at compile time, so a runtime stride would simply not match the
// VectorLayout specializations. The frontend therefore refuses a VectorLayout
// tile whose stride is anything but a static 1; a strided rank-1 view is spelled
// RowMajor, whose descriptor keeps the real pitch.
CATLASS_DEVICE auto makeVectorTlaLayout(const TensorDesc& desc)
{
    return tla::MakeLayout(
        tla::MakeShape(desc.shape1), tla::MakeStride(tla::Int<1>{}), tla::MakeShape(desc.originShape1));
}

CATLASS_DEVICE int64_t makeVectorTlaCoord(const TensorDesc& desc)
{
    return static_cast<int64_t>(desc.coord0) * desc.stride0 + static_cast<int64_t>(desc.coord1);
}

template <typename T>
CATLASS_DEVICE auto makezNTlaLayout(const TensorDesc& desc)
{
    constexpr uint32_t eleNumPerC0 = Catlass::BytesToBits(Catlass::BYTE_PER_C0) / Catlass::SizeOfBits<T>::value;
    constexpr uint32_t eleNumPerFractal =
        Catlass::BytesToBits(Catlass::BYTE_PER_FRACTAL) / Catlass::SizeOfBits<T>::value;
    return tla::MakeLayout(
        tla::MakeShape(
            tla::MakeShape(tla::Int<Catlass::C0_NUM_PER_FRACTAL>{}, desc.shape1),
            tla::MakeShape(tla::Int<eleNumPerC0>{}, desc.shape3)),
        tla::MakeStride(
            tla::MakeStride(tla::Int<eleNumPerC0>{}, tla::Int<eleNumPerFractal>{}),
            tla::MakeStride(tla::Int<1>{}, desc.stride3)),
        tla::MakeShape(desc.originShape0, desc.originShape1));
}

template <typename T>
CATLASS_DEVICE auto makezNUnAlignTlaLayout(const TensorDesc& desc)
{
    constexpr uint32_t eleNumPerC0 = Catlass::BytesToElements<T>(Catlass::BYTE_PER_C0);
    constexpr uint32_t eleNumPerFractal = Catlass::BytesToElements<T>(Catlass::BYTE_PER_FRACTAL);
    return tla::MakeLayout(
        tla::MakeShape(
            tla::MakeShape(desc.shape0 * desc.shape1, tla::Int<1>{}),
            tla::MakeShape(tla::Int<eleNumPerC0>{}, desc.shape3)),
        tla::MakeStride(
            tla::MakeStride(tla::Int<eleNumPerC0>{}, desc.stride3), tla::MakeStride(tla::Int<1>{}, desc.stride3)),
        tla::MakeShape(desc.originShape0, desc.originShape1));
}

template <typename T>
CATLASS_DEVICE auto makenZTlaLayout(const TensorDesc& desc)
{
    constexpr uint32_t eleNumPerC0 = Catlass::BytesToBits(Catlass::BYTE_PER_C0) / Catlass::SizeOfBits<T>::value;
    constexpr uint32_t eleNumPerFractal =
        Catlass::BytesToBits(Catlass::BYTE_PER_FRACTAL) / Catlass::SizeOfBits<T>::value;
    return tla::MakeLayout(
        tla::MakeShape(
            tla::MakeShape(tla::Int<eleNumPerC0>{}, desc.shape1),
            tla::MakeShape(tla::Int<Catlass::C0_NUM_PER_FRACTAL>{}, desc.shape3)),
        tla::MakeStride(
            tla::MakeStride(tla::Int<1>{}, desc.stride1),
            tla::MakeStride(tla::Int<eleNumPerC0>{}, tla::Int<eleNumPerFractal>{})),
        tla::MakeShape(desc.originShape0, desc.originShape1));
}

template <typename T>
CATLASS_DEVICE auto makeL0CTlaLayout(const TensorDesc& desc)
{
    constexpr uint32_t eleNumPerFractal = 256;
    return tla::MakeLayout(
        tla::MakeShape(
            tla::MakeShape(tla::Int<Catlass::C0_NUM_PER_FRACTAL>{}, desc.shape1),
            tla::MakeShape(tla::Int<Catlass::C0_NUM_PER_FRACTAL>{}, desc.shape3)),
        tla::MakeStride(
            tla::MakeStride(tla::Int<Catlass::C0_NUM_PER_FRACTAL>{}, tla::Int<eleNumPerFractal>{}),
            tla::MakeStride(tla::Int<1>{}, desc.stride3)),
        tla::MakeShape(desc.originShape0, desc.originShape1));
}

// MX shared-exponent scale block, laid out like zZ but with the C0 fixed by the
// e8m0 format (2 elements per C0, 32 bytes per fractal) rather than derived from
// the element width. Mirrors tla::MakeMxScaleLayout<..., zZ, ...>; the descriptor
// already carries the nested tile shape and parent strides, as for zN/nZ.
CATLASS_DEVICE auto makeMxScalezZTlaLayout(const TensorDesc& desc)
{
    return tla::MakeLayout(
        tla::MakeShape(
            tla::MakeShape(tla::Int<Catlass::C0_NUM_PER_FRACTAL>{}, desc.shape1),
            tla::MakeShape(tla::Int<2>{}, desc.shape3)),
        tla::MakeStride(tla::MakeStride(tla::Int<2>{}, desc.stride1), tla::MakeStride(tla::Int<1>{}, tla::Int<32>{})),
        tla::MakeShape(desc.originShape0, desc.originShape1));
}

// The B-side scale block: same idea, transposed nesting (copy_l1_to_l0b requires
// nN where copy_l1_to_l0a requires zZ).
CATLASS_DEVICE auto makeMxScalenNTlaLayout(const TensorDesc& desc)
{
    return tla::MakeLayout(
        tla::MakeShape(
            tla::MakeShape(tla::Int<2>{}, desc.shape1),
            tla::MakeShape(tla::Int<Catlass::C0_NUM_PER_FRACTAL>{}, desc.shape3)),
        tla::MakeStride(tla::MakeStride(tla::Int<1>{}, tla::Int<32>{}), tla::MakeStride(tla::Int<2>{}, desc.stride3)),
        tla::MakeShape(desc.originShape0, desc.originShape1));
}

// ---------------------------------------------------------------------------
// Position-tagged tensor constructors. LayoutTag is a non-type template
// parameter; each Position selects the layout constructor appropriate to that
// memory (e.g. UB zN uses the UnAlign variant, L1/L0A zN uses the aligned one).
// ---------------------------------------------------------------------------

template <LayoutTag Tag, typename T>
CATLASS_DEVICE auto makeGMTensor(
    memref_t<__gm__ T, 2>* memref, const TensorDesc& desc, int64_t cachePolicy = kCachePolicyNormal)
{
    if constexpr (Tag == LayoutTag::RowMajor) {
        return tla::MakeTensor(
            makeGlobalTensor(basePtr(memref), cachePolicy), makeRowMajorTlaLayout<T>(desc), makeTlaTileCoord(desc),
            Catlass::Arch::PositionGM{});
    } else if constexpr (Tag == LayoutTag::ColumnMajor) {
        return tla::MakeTensor(
            makeGlobalTensor(basePtr(memref), cachePolicy), makeColumnMajorTlaLayout(desc), makeTlaTileCoord(desc),
            Catlass::Arch::PositionGM{});
    } else if constexpr (Tag == LayoutTag::VectorLayout) {
        // VectorLayout is the flat tag: rank-1, so the VectorLayout
        // TileCopyTla specialization matches it directly.
        return tla::MakeTensor(
            makeGlobalTensor(basePtr(memref), cachePolicy), makeVectorTlaLayout(desc),
            tla::MakeCoord(makeVectorTlaCoord(desc)), Catlass::Arch::PositionGM{});
    } else if constexpr (Tag == LayoutTag::zN) {
        // A GM tile that already sits in zN fractal order. The layout constructor
        // is the same one the L1 side uses, so the copy is a plain block move
        // rather than an ND -> zN re-layout.
        //
        // This is the tag for a prepacked weight buffer. It is not the tight
        // packing of a partial row count: a GM buffer read as zN is padded to
        // whole 16-row fractals at both ends, so its row pitch is
        // ceil(M/16) * 16 * C0 and there is no second tag to choose between here.
        // A tight M axis belongs to the UB -> L1 route, where the source tile is
        // built in UB and its pitch is the vector unit's business.
        return tla::MakeTensor(
            makeGlobalTensor(basePtr(memref), cachePolicy), makezNTlaLayout<T>(desc), makeTlaTileCoord(desc),
            Catlass::Arch::PositionGM{});
    } else if constexpr (Tag == LayoutTag::nZ) {
        // The transposed mirror: the GM tile already sits in the (C0-row,
        // 16-column) fractal order the L1 nZ tile uses, so again the same layout
        // constructor describes both ends and the copy reorders nothing. The
        // row-group pitch is desc.stride1 (cols rounded up to 16 times C0) and
        // the column-group pitch is the constant fractal size, both of which the
        // host-side layout already emits -- see the nZ branch of the frontend's
        // layout trees.
        //
        // nZ has no unaligned variant at all, and both ends must be allocated for
        // the padded tile (ceil(M/C0) * C0 rows by ceil(N/16) * 16 columns),
        // because the copy steps whole row groups and the last one moves its full
        // C0 rows.
        return tla::MakeTensor(
            makeGlobalTensor(basePtr(memref), cachePolicy), makenZTlaLayout<T>(desc), makeTlaTileCoord(desc),
            Catlass::Arch::PositionGM{});
    } else if constexpr (Tag == LayoutTag::zNUnAlign) {
        // A feature map in NC1HWC0: ((hw, 1), (C0, c1)) with the c1 leaf at the
        // full-feature-map pitch. Same tree as the L1 tile it is copied into, so
        // the copy is a run of contiguous C1 slices rather than a re-layout.
        //
        // This is a separate route from the zN block move above -- it feeds
        // tla.experimental.img2col and lands on an L1 zNUnAlign tile, not on an
        // L1 zN one -- which is why both tags exist on the GM side.
        return tla::MakeTensor(
            makeGlobalTensor(basePtr(memref), cachePolicy), makezNUnAlignTlaLayout<T>(desc), makeTlaTileCoord(desc),
            Catlass::Arch::PositionGM{});
    } else if constexpr (Tag == LayoutTag::RowMajorMxScaleA) {
        // GM MX scale, one tag per (side, orientation). The desc's 2x2 NZFamily
        // tree is re-packed so Catlass's scale predicates see the original ranks.
        static_assert(std::is_same_v<T, float8_e8m0_t>, "an MX scale tile is float8_e8m0_t");
        // desc: shape (1, rows, C0, colGroups); stride (0, pitch, 1, C0).
        return tla::MakeTensor(
            makeGlobalTensor(basePtr(memref), cachePolicy),
            tla::MakeLayout(
                tla::MakeShape(desc.shape1, tla::MakeShape(tla::Int<2>{}, desc.shape3)),
                tla::MakeStride(desc.stride1, tla::MakeStride(tla::Int<1>{}, tla::Int<2>{})),
                tla::MakeShape(desc.originShape0, desc.originShape1)),
            makeTlaTileCoord(desc), Catlass::Arch::PositionGM{});
    } else if constexpr (Tag == LayoutTag::ColMajorMxScaleA) {
        static_assert(std::is_same_v<T, float8_e8m0_t>, "an MX scale tile is float8_e8m0_t");
        // desc: shape (1, rows, C0, colGroups); stride (0, C0, 1, rows*C0).
        // ND2NZ interleaves groups in C0 pairs, so the row stride is literal C0.
        return tla::MakeTensor(
            makeGlobalTensor(basePtr(memref), cachePolicy),
            tla::MakeLayout(
                tla::MakeShape(desc.shape1, tla::MakeShape(tla::Int<2>{}, desc.shape3)),
                tla::MakeStride(tla::Int<2>{}, tla::MakeStride(tla::Int<1>{}, desc.stride3)),
                tla::MakeShape(desc.originShape0, desc.originShape1)),
            makeTlaTileCoord(desc), Catlass::Arch::PositionGM{});
    } else if constexpr (Tag == LayoutTag::RowMajorMxScaleB) {
        static_assert(std::is_same_v<T, float8_e8m0_t>, "an MX scale tile is float8_e8m0_t");
        // desc: shape (C0, rowGroups, 1, cols); stride (1, cols*C0, 0, C0).
        return tla::MakeTensor(
            makeGlobalTensor(basePtr(memref), cachePolicy),
            tla::MakeLayout(
                tla::MakeShape(tla::MakeShape(tla::Int<2>{}, desc.shape1), desc.shape3),
                tla::MakeStride(tla::MakeStride(tla::Int<1>{}, desc.stride1), tla::Int<2>{}),
                tla::MakeShape(desc.originShape0, desc.originShape1)),
            makeTlaTileCoord(desc), Catlass::Arch::PositionGM{});
    } else if constexpr (Tag == LayoutTag::ColMajorMxScaleB) {
        static_assert(std::is_same_v<T, float8_e8m0_t>, "an MX scale tile is float8_e8m0_t");
        // desc: shape (C0, rowGroups, 1, cols); stride (1, C0, 0, pitch).
        return tla::MakeTensor(
            makeGlobalTensor(basePtr(memref), cachePolicy),
            tla::MakeLayout(
                tla::MakeShape(tla::MakeShape(tla::Int<2>{}, desc.shape1), desc.shape3),
                tla::MakeStride(tla::MakeStride(tla::Int<1>{}, tla::Int<2>{}), desc.stride3),
                tla::MakeShape(desc.originShape0, desc.originShape1)),
            makeTlaTileCoord(desc), Catlass::Arch::PositionGM{});
    } else {
        static_assert(
            Tag == LayoutTag::RowMajor || Tag == LayoutTag::ColumnMajor || Tag == LayoutTag::VectorLayout ||
                Tag == LayoutTag::zN || Tag == LayoutTag::nZ || Tag == LayoutTag::zNUnAlign,
            "GM tensor supports RowMajor/ColumnMajor/VectorLayout/zN/nZ/zNUnAlign and "
            "RowMajor/ColMajor MxScale A/B for float8_e8m0_t only");
    }
}

template <LayoutTag Tag, typename T>
CATLASS_DEVICE auto makeUBTensor(memref_t<__ubuf__ T, 1>* memref, const TensorDesc& desc)
{
    AscendC::LocalTensor<T> tensor(AscendC::TPosition::VECCALC, localAddr(memref), elementCount(memref));
    if constexpr (Tag == LayoutTag::RowMajor) {
        return tla::MakeTensor(
            tensor, makeRowMajorTlaLayout<T>(desc), makeTlaTileCoord(desc), Catlass::Arch::PositionUB{});
    } else if constexpr (Tag == LayoutTag::ColumnMajor) {
        return tla::MakeTensor(
            tensor, makeColumnMajorTlaLayout(desc), makeTlaTileCoord(desc), Catlass::Arch::PositionUB{});
    } else if constexpr (Tag == LayoutTag::zN || Tag == LayoutTag::zNUnAlign) {
        return tla::MakeTensor(
            tensor, makezNUnAlignTlaLayout<T>(desc), makeTlaTileCoord(desc), Catlass::Arch::PositionUB{});
    } else if constexpr (Tag == LayoutTag::zZMxScale) {
        static_assert(std::is_same_v<T, float8_e8m0_t>, "an MX scale tile is float8_e8m0_t");
        return tla::MakeTensor(
            tensor, makeMxScalezZTlaLayout(desc), makeTlaTileCoord(desc), Catlass::Arch::PositionUB{});
    } else if constexpr (Tag == LayoutTag::nNMxScale) {
        static_assert(std::is_same_v<T, float8_e8m0_t>, "an MX scale tile is float8_e8m0_t");
        return tla::MakeTensor(
            tensor, makeMxScalenNTlaLayout(desc), makeTlaTileCoord(desc), Catlass::Arch::PositionUB{});
    } else {
        static_assert(
            Tag == LayoutTag::RowMajor || Tag == LayoutTag::ColumnMajor || Tag == LayoutTag::zN ||
                Tag == LayoutTag::zNUnAlign,
            "UB tensor supports RowMajor/ColumnMajor/zN and zZMxScale/nNMxScale for float8_e8m0_t only");
    }
}

template <LayoutTag Tag, typename T>
CATLASS_DEVICE auto makeL1Tensor(memref_t<__cbuf__ T, 1>* memref, const TensorDesc& desc)
{
    AscendC::LocalTensor<T> tensor(AscendC::TPosition::A1, localAddr(memref), elementCount(memref));
    if constexpr (Tag == LayoutTag::zN) {
        return tla::MakeTensor(tensor, makezNTlaLayout<T>(desc), makeTlaTileCoord(desc), Catlass::Arch::PositionL1{});
    } else if constexpr (Tag == LayoutTag::nZ) {
        return tla::MakeTensor(tensor, makenZTlaLayout<T>(desc), makeTlaTileCoord(desc), Catlass::Arch::PositionL1{});
    } else if constexpr (Tag == LayoutTag::VectorLayout) {
        // A row-major L1 tile is one contiguous run of elements, so it is built
        // rank-1: the VectorLayout TileCopyTla specializations match on that
        // shape, and fixpipe reads the same base offset through layout()(coord).
        // VectorLayout says the same thing as a tag instead of by shape, and needs the
        // identical construction.
        return tla::MakeTensor(
            tensor, makeVectorTlaLayout(desc), tla::MakeCoord(makeVectorTlaCoord(desc)), Catlass::Arch::PositionL1{});
    } else if constexpr (Tag == LayoutTag::zNUnAlign) {
        // The NC1HWC0 feature-map tile that tla.img2col reads. Its M axis is not
        // fractal-blocked -- h*w rows sit back to back inside each C1 slice --
        // which is exactly what zNUnAlign already describes.
        return tla::MakeTensor(
            tensor, makezNUnAlignTlaLayout<T>(desc), makeTlaTileCoord(desc), Catlass::Arch::PositionL1{});
    } else if constexpr (Tag == LayoutTag::zZMxScale) {
        static_assert(std::is_same_v<T, float8_e8m0_t>, "an MX scale tile is float8_e8m0_t");
        return tla::MakeTensor(
            tensor, makeMxScalezZTlaLayout(desc), makeTlaTileCoord(desc), Catlass::Arch::PositionL1{});
    } else if constexpr (Tag == LayoutTag::nNMxScale) {
        static_assert(std::is_same_v<T, float8_e8m0_t>, "an MX scale tile is float8_e8m0_t");
        return tla::MakeTensor(
            tensor, makeMxScalenNTlaLayout(desc), makeTlaTileCoord(desc), Catlass::Arch::PositionL1{});
    } else {
        static_assert(
            Tag == LayoutTag::zN || Tag == LayoutTag::nZ || Tag == LayoutTag::VectorLayout ||
                Tag == LayoutTag::zNUnAlign,
            "L1 tensor supports zN/nZ/VectorLayout/zNUnAlign and zZMxScale/nNMxScale for float8_e8m0_t only");
    }
}

template <LayoutTag Tag, typename T>
CATLASS_DEVICE auto makeL0ATensor(memref_t<__ca__ T, 1>* memref, const TensorDesc& desc)
{
    static_assert(Tag == LayoutTag::zN, "L0A tensor supports zN only");
    AscendC::LocalTensor<T> tensor(AscendC::TPosition::A2, localAddr(memref), elementCount(memref));
    return tla::MakeTensor(tensor, makezNTlaLayout<T>(desc), makeTlaTileCoord(desc), Catlass::Arch::PositionL0A{});
}

template <LayoutTag Tag, typename T>
CATLASS_DEVICE auto makeL0BTensor(memref_t<__cb__ T, 1>* memref, const TensorDesc& desc)
{
    static_assert(Tag == LayoutTag::nZ, "L0B tensor supports nZ only");
    AscendC::LocalTensor<T> tensor(AscendC::TPosition::B2, localAddr(memref), elementCount(memref));
    return tla::MakeTensor(tensor, makenZTlaLayout<T>(desc), makeTlaTileCoord(desc), Catlass::Arch::PositionL0B{});
}

template <LayoutTag Tag, typename T>
CATLASS_DEVICE auto makeBiasTensor(memref_t<__biasbuf__ T, 1>* memref, const TensorDesc& desc)
{
    static_assert(Tag == LayoutTag::VectorLayout, "Bias tensor supports VectorLayout only");
    AscendC::LocalTensor<T> tensor(AscendC::TPosition::C2, localAddr(memref), elementCount(memref));
    return tla::MakeTensor(
        tensor, makeVectorTlaLayout(desc), tla::MakeCoord(makeVectorTlaCoord(desc)), Catlass::Arch::PositionBias{});
}

template <LayoutTag Tag, typename T>
CATLASS_DEVICE auto makeL0CTensor(memref_t<__cc__ T, 1>* memref, const TensorDesc& desc)
{
    static_assert(Tag == LayoutTag::L0Clayout, "L0C tensor supports L0C only");
    AscendC::LocalTensor<T> tensor(AscendC::TPosition::CO1, localAddr(memref), elementCount(memref));
    return tla::MakeTensor(tensor, makeL0CTlaLayout<T>(desc), makeTlaTileCoord(desc), Catlass::Arch::PositionL0C{});
}

// ---------------------------------------------------------------------------
// C-ABI helpers for the unified 12-field descriptor. DESC_ABI_PARAMS declares
// the 12 int64 descriptor args in a ciface signature; TENSOR_DESC_12 builds the
// matching TensorDesc from them. Both take a prefix token (src/dst).
// ---------------------------------------------------------------------------
#define DESC_ABI_PARAMS(P)                                                                                \
    int64_t P##Shape0, int64_t P##Shape1, int64_t P##Shape2, int64_t P##Shape3, int64_t P##Stride0,       \
        int64_t P##Stride1, int64_t P##Stride2, int64_t P##Stride3, int64_t P##Coord0, int64_t P##Coord1, \
        int64_t P##OrgShape0, int64_t P##OrgShape1

#define TENSOR_DESC_12(P)                                                                                            \
    TensorDesc                                                                                                       \
    {                                                                                                                \
        (uint32_t) P##Shape0, (uint32_t)P##Shape1, (uint32_t)P##Shape2, (uint32_t)P##Shape3, P##Stride0, P##Stride1, \
            P##Stride2, P##Stride3, (uint32_t)P##Coord0, (uint32_t)P##Coord1, (uint32_t)P##OrgShape0,                \
            (uint32_t)P##OrgShape1                                                                                   \
    }
