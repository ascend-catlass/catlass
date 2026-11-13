#pragma once

#include "common.h"

#include "catlass/arch/arch.hpp"
#include "catlass/layout/layout.hpp"
#include "catlass/numeric_size.hpp"
#include "tla/layout.hpp"
#include "tla/tensor.hpp"

using bf16 = bfloat16_t;

#if ((defined(__NPU_ARCH__) && __NPU_ARCH__ == 3510) || (defined(CATLASS_ARCH) && CATLASS_ARCH == 3510))
// The MX fp8 operand types live in the AscendC namespace, and the REGISTER_*
// macros paste their argument into both the symbol name and the signature -- a
// `::` cannot go into a symbol. Re-declare them at global scope under the *same*
// spelling so the token stays the real C++ type name, which is what every bc
// symbol suffix is. The fp4 types need no alias: float4_e2m1x2_t is already a
// global name.
using mx_fp8_e4m3_t = AscendC::mx_fp8_e4m3_t;
using mx_fp8_e5m2_t = AscendC::mx_fp8_e5m2_t;
#endif

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

template <typename T>
CATLASS_DEVICE AscendC::GlobalTensor<T> makeGlobalTensor(__gm__ T* ptr)
{
    AscendC::GlobalTensor<T> tensor;
    tensor.SetGlobalBuffer(ptr);
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
    constexpr uint32_t eleNumPerC0 = Catlass::BYTE_PER_C0 / sizeof(T);
    constexpr uint32_t eleNumPerFractal = Catlass::BYTE_PER_FRACTAL / sizeof(T);
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

// ---------------------------------------------------------------------------
// Position-tagged tensor constructors. LayoutTag is a non-type template
// parameter; each Position selects the layout constructor appropriate to that
// memory (e.g. UB zN uses the UnAlign variant, L1/L0A zN uses the aligned one).
// ---------------------------------------------------------------------------

template <LayoutTag Tag, typename T>
CATLASS_DEVICE auto makeGMTensor(memref_t<__gm__ T, 2>* memref, const TensorDesc& desc)
{
    if constexpr (Tag == LayoutTag::RowMajor) {
        return tla::MakeTensor(
            makeGlobalTensor(basePtr(memref)), makeRowMajorTlaLayout<T>(desc), makeTlaTileCoord(desc),
            Catlass::Arch::PositionGM{});
    } else if constexpr (Tag == LayoutTag::ColumnMajor) {
        return tla::MakeTensor(
            makeGlobalTensor(basePtr(memref)), makeColumnMajorTlaLayout(desc), makeTlaTileCoord(desc),
            Catlass::Arch::PositionGM{});
    } else if constexpr (Tag == LayoutTag::VectorLayout) {
        // VectorLayout is the flat tag: rank-1, so the VectorLayout
        // TileCopyTla specialization matches it directly.
        return tla::MakeTensor(
            makeGlobalTensor(basePtr(memref)), makeVectorTlaLayout(desc), tla::MakeCoord(makeVectorTlaCoord(desc)),
            Catlass::Arch::PositionGM{});
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
            makeGlobalTensor(basePtr(memref)), makezNTlaLayout<T>(desc), makeTlaTileCoord(desc),
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
            makeGlobalTensor(basePtr(memref)), makenZTlaLayout<T>(desc), makeTlaTileCoord(desc),
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
            makeGlobalTensor(basePtr(memref)), makezNUnAlignTlaLayout<T>(desc), makeTlaTileCoord(desc),
            Catlass::Arch::PositionGM{});
    } else {
        static_assert(
            Tag == LayoutTag::RowMajor || Tag == LayoutTag::ColumnMajor || Tag == LayoutTag::VectorLayout ||
                Tag == LayoutTag::zN || Tag == LayoutTag::nZ || Tag == LayoutTag::zNUnAlign,
            "GM tensor supports RowMajor/ColumnMajor/VectorLayout/zN/nZ/zNUnAlign only");
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
    } else {
        static_assert(
            Tag == LayoutTag::RowMajor || Tag == LayoutTag::ColumnMajor || Tag == LayoutTag::zN ||
                Tag == LayoutTag::zNUnAlign,
            "UB tensor supports RowMajor/ColumnMajor/zN only");
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
    } else {
        static_assert(
            Tag == LayoutTag::zN || Tag == LayoutTag::nZ || Tag == LayoutTag::VectorLayout ||
                Tag == LayoutTag::zNUnAlign,
            "L1 tensor supports zN/nZ/VectorLayout/zNUnAlign only");
    }
}

#if ((defined(__NPU_ARCH__) && __NPU_ARCH__ == 3510) || (defined(CATLASS_ARCH) && CATLASS_ARCH == 3510))
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

// GM-side MX scale tensors, one per (side, orientation).
//
// The IR descriptor now carries the NZFamily 2x2 tree -- shape
// ((1, rows), (C0, colGroups)) or ((C0, rowGroups), (1, cols)), with the
// leading (1, dim) pair's first leaf at stride 0 -- following
// tla::MakeMxScaleLayout with the rank-1 extent made 2-D. Catlass's copy
// predicates (isMxScaleForRowMajorA / ColumnMajorA / RowMajorB / ColumnMajorB)
// key on the *original* rank structure, where the plain matrix dimension is a
// scalar leaf, so this converts back to that form: the 4 desc leaves name the
// same rows / groups / pitch values the old reconstruction used, just at
// different slots.
template <LayoutTag Tag>
CATLASS_DEVICE auto makeGMMxScaleTensor(memref_t<__gm__ uint8_t, 2>* memref, const TensorDesc& desc)
{
    auto g = makeGlobalTensor(reinterpret_cast<__gm__ float8_e8m0_t*>(basePtr(memref)));
    constexpr uint32_t C0 = 2; // e8m0 scale C0, fixed by the format
    const auto origin = tla::MakeShape(desc.originShape0, desc.originShape1);
    const auto coord = makeTlaTileCoord(desc);

    if constexpr (Tag == LayoutTag::RowMajorMxScaleA) {
        // desc: shape (1, rows, C0, colGroups); stride (0, pitch, 1, C0).
        return tla::MakeTensor(
            g,
            tla::MakeLayout(
                tla::MakeShape(desc.shape1, tla::MakeShape(tla::Int<C0>{}, desc.shape3)),
                tla::MakeStride(desc.stride1, tla::MakeStride(tla::Int<1>{}, tla::Int<C0>{})), origin),
            coord, Catlass::Arch::PositionGM{});
    } else if constexpr (Tag == LayoutTag::ColMajorMxScaleA) {
        // desc: shape (1, rows, C0, colGroups); stride (0, C0, 1, rows*C0).
        // Catlass reaches this one through ND2NZ where row-major A uses the
        // transposing DN2NZ, so the GM block arrives with its groups
        // interleaved in C0-sized pairs and the row index running fastest
        // between pairs; the row stride is the literal C0 -- which is also
        // what the isMxScaleForColumnMajorA predicate keys on.
        return tla::MakeTensor(
            g,
            tla::MakeLayout(
                tla::MakeShape(desc.shape1, tla::MakeShape(tla::Int<C0>{}, desc.shape3)),
                tla::MakeStride(tla::Int<C0>{}, tla::MakeStride(tla::Int<1>{}, desc.stride3)), origin),
            coord, Catlass::Arch::PositionGM{});
    } else if constexpr (Tag == LayoutTag::RowMajorMxScaleB) {
        // desc: shape (C0, rowGroups, 1, cols); stride (1, cols*C0, 0, C0).
        // The mirror of column-major A above, and ND2NZ for the same reason.
        return tla::MakeTensor(
            g,
            tla::MakeLayout(
                tla::MakeShape(tla::MakeShape(tla::Int<C0>{}, desc.shape1), desc.shape3),
                tla::MakeStride(tla::MakeStride(tla::Int<1>{}, desc.stride1), tla::Int<C0>{}), origin),
            coord, Catlass::Arch::PositionGM{});
    } else {
        static_assert(
            Tag == LayoutTag::ColMajorMxScaleB,
            "GM MX scale supports RowMajorMxScaleA / ColMajorMxScaleA / RowMajorMxScaleB / ColMajorMxScaleB");
        // desc: shape (C0, rowGroups, 1, cols); stride (1, C0, 0, pitch).
        return tla::MakeTensor(
            g,
            tla::MakeLayout(
                tla::MakeShape(tla::MakeShape(tla::Int<C0>{}, desc.shape1), desc.shape3),
                tla::MakeStride(tla::MakeStride(tla::Int<1>{}, tla::Int<C0>{}), desc.stride3), origin),
            coord, Catlass::Arch::PositionGM{});
    }
}

// The UB counterpart, for a scale block produced on the vector unit. Same ABI
// (u8) and the same restored e8m0 element type; only the position differs.
template <LayoutTag Tag>
CATLASS_DEVICE auto makeUBMxScaleTensor(memref_t<__ubuf__ uint8_t, 1>* memref, const TensorDesc& desc)
{
    AscendC::LocalTensor<float8_e8m0_t> tensor(AscendC::TPosition::VECCALC, localAddr(memref), elementCount(memref));
    if constexpr (Tag == LayoutTag::zZMxScale) {
        return tla::MakeTensor(
            tensor, makeMxScalezZTlaLayout(desc), makeTlaTileCoord(desc), Catlass::Arch::PositionUB{});
    } else {
        static_assert(Tag == LayoutTag::nNMxScale, "MX scale tile supports zZMxScale (A) / nNMxScale (B) only");
        return tla::MakeTensor(
            tensor, makeMxScalenNTlaLayout(desc), makeTlaTileCoord(desc), Catlass::Arch::PositionUB{});
    }
}

// The scale block travels over the ABI as u8 (MLIR has no e8m0 type); the
// element type is restored here, where the copy templates require it.
template <LayoutTag Tag>
CATLASS_DEVICE auto makeL1MxScaleTensor(memref_t<__cbuf__ uint8_t, 1>* memref, const TensorDesc& desc)
{
    AscendC::LocalTensor<float8_e8m0_t> tensor(AscendC::TPosition::A1, localAddr(memref), elementCount(memref));
    if constexpr (Tag == LayoutTag::zZMxScale) {
        return tla::MakeTensor(
            tensor, makeMxScalezZTlaLayout(desc), makeTlaTileCoord(desc), Catlass::Arch::PositionL1{});
    } else {
        static_assert(Tag == LayoutTag::nNMxScale, "MX scale tile supports zZMxScale (A) / nNMxScale (B) only");
        return tla::MakeTensor(
            tensor, makeMxScalenNTlaLayout(desc), makeTlaTileCoord(desc), Catlass::Arch::PositionL1{});
    }
}
#endif

// Packed fp4 tiles travel over the bc ABI as int8_t storage -- the ABI is
// byte-denominated, whatever the tile's element type says -- and the real
// element type is restored here, where Catlass's SizeOfBits<> gives the layout
// math its 4-bit element width. Shapes in the descriptor are in fp4 elements, so
// the byte extent is half of that.
// zN and nZ produce distinct Tensor types, so the choice must be a template
// parameter, not a runtime flag. It is spelled as a LayoutTag rather than a bool
// so the call sites read as the layout they mean, like every other tensor
// builder here.
template <typename TFp4, LayoutTag Tag>
CATLASS_DEVICE auto makeL1Fp4Tensor(memref_t<__cbuf__ int8_t, 1>* memref, const TensorDesc& desc)
{
    static_assert(Tag == LayoutTag::zN || Tag == LayoutTag::nZ, "L1 fp4 tile supports zN / nZ only");
    AscendC::LocalTensor<TFp4> tensor(AscendC::TPosition::A1, localAddr(memref), elementCount(memref));
    if constexpr (Tag == LayoutTag::nZ) {
        return tla::MakeTensor(
            tensor, makenZTlaLayout<TFp4>(desc), makeTlaTileCoord(desc), Catlass::Arch::PositionL1{});
    } else {
        return tla::MakeTensor(
            tensor, makezNTlaLayout<TFp4>(desc), makeTlaTileCoord(desc), Catlass::Arch::PositionL1{});
    }
}

// The UB counterpart of makeL1Fp4Tensor. Same reasoning: the ABI hands the tile
// over as bytes, and restoring TFp4 here gives the layout math its 4-bit
// element width, so a descriptor counted in fp4 elements spans half as many
// bytes.
template <typename TFp4, LayoutTag Tag>
CATLASS_DEVICE auto makeUBFp4Tensor(memref_t<__ubuf__ int8_t, 1>* memref, const TensorDesc& desc)
{
    static_assert(Tag == LayoutTag::RowMajor || Tag == LayoutTag::zN, "UB fp4 tile supports RowMajor / zN only");
    AscendC::LocalTensor<TFp4> tensor(AscendC::TPosition::VECCALC, localAddr(memref), elementCount(memref));
    if constexpr (Tag == LayoutTag::RowMajor) {
        return tla::MakeTensor(
            tensor, makeRowMajorTlaLayout<TFp4>(desc), makeTlaTileCoord(desc), Catlass::Arch::PositionUB{});
    } else {
        return tla::MakeTensor(
            tensor, makezNTlaLayout<TFp4>(desc), makeTlaTileCoord(desc), Catlass::Arch::PositionUB{});
    }
}

template <typename TFp4, LayoutTag Tag>
CATLASS_DEVICE auto makeGMFp4Tensor(memref_t<__gm__ int8_t, 2>* memref, const TensorDesc& desc)
{
    static_assert(
        Tag == LayoutTag::RowMajor || Tag == LayoutTag::ColumnMajor,
        "GM fp4 tile supports RowMajor / ColumnMajor only");
    auto g = makeGlobalTensor(reinterpret_cast<__gm__ TFp4*>(basePtr(memref)));
    if constexpr (Tag == LayoutTag::ColumnMajor) {
        return tla::MakeTensor(g, makeColumnMajorTlaLayout(desc), makeTlaTileCoord(desc), Catlass::Arch::PositionGM{});
    } else {
        return tla::MakeTensor(
            g, makeRowMajorTlaLayout<TFp4>(desc), makeTlaTileCoord(desc), Catlass::Arch::PositionGM{});
    }
}

// TElem is allowed to differ from the memref's storage type: an MX L0 tile is
// addressed as mx_fp8_* / float4_*x2_t while the ABI hands it over as the byte
// storage those occupy. The width is the same either way, so only the
// LocalTensor's element type changes and the layout math is untouched.
template <LayoutTag Tag, typename TElem, typename TStore>
CATLASS_DEVICE auto makeL0ATensorAs(memref_t<__ca__ TStore, 1>* memref, const TensorDesc& desc)
{
    static_assert(Tag == LayoutTag::zN, "L0A tensor supports zN only");
    AscendC::LocalTensor<TElem> tensor(AscendC::TPosition::A2, localAddr(memref), elementCount(memref));
    return tla::MakeTensor(tensor, makezNTlaLayout<TElem>(desc), makeTlaTileCoord(desc), Catlass::Arch::PositionL0A{});
}

template <LayoutTag Tag, typename TElem, typename TStore>
CATLASS_DEVICE auto makeL0BTensorAs(memref_t<__cb__ TStore, 1>* memref, const TensorDesc& desc)
{
    static_assert(Tag == LayoutTag::nZ, "L0B tensor supports nZ only");
    AscendC::LocalTensor<TElem> tensor(AscendC::TPosition::B2, localAddr(memref), elementCount(memref));
    return tla::MakeTensor(tensor, makenZTlaLayout<TElem>(desc), makeTlaTileCoord(desc), Catlass::Arch::PositionL0B{});
}

template <LayoutTag Tag, typename T>
CATLASS_DEVICE auto makeL0ATensor(memref_t<__ca__ T, 1>* memref, const TensorDesc& desc)
{
    return makeL0ATensorAs<Tag, T, T>(memref, desc);
}

template <LayoutTag Tag, typename T>
CATLASS_DEVICE auto makeL0BTensor(memref_t<__cb__ T, 1>* memref, const TensorDesc& desc)
{
    return makeL0BTensorAs<Tag, T, T>(memref, desc);
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
