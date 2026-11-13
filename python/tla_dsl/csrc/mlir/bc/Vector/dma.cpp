#include "../dma_common.h"

#include "catlass/epilogue/tile/tile_copy.hpp"

// The MX block scale UB -> L1 specialization, kept out of the shared CATLASS
// header because it is a DSL-side concept. Must precede the instantiations below.
#include "mx_copy_ub_to_l1.h"

template <class ArchTag, LayoutTag SrcLayout, LayoutTag DstLayout, typename T>
CATLASS_DEVICE void copyGMToUB(
    memref_t<__gm__ T, 2>* src, memref_t<__ubuf__ T, 1>* dst, const TensorDesc& srcDesc, const TensorDesc& dstDesc,
    int64_t gmCachePolicy = kCachePolicyNormal)
{
    auto srcTensor = makeGMTensor<SrcLayout, T>(src, srcDesc, gmCachePolicy);
    auto dstTensor = makeUBTensor<DstLayout, T>(dst, dstDesc);
    Catlass::Epilogue::Tile::CopyGm2UbTla<ArchTag, decltype(srcTensor), decltype(dstTensor)>{}(dstTensor, srcTensor);
}

template <class ArchTag, LayoutTag SrcLayout, LayoutTag DstLayout, typename T>
CATLASS_DEVICE void copyUBToGM(
    memref_t<__ubuf__ T, 1>* src, memref_t<__gm__ T, 2>* dst, const TensorDesc& srcDesc, const TensorDesc& dstDesc,
    int64_t gmCachePolicy = kCachePolicyNormal)
{
    auto srcTensor = makeUBTensor<SrcLayout, T>(src, srcDesc);
    auto dstTensor = makeGMTensor<DstLayout, T>(dst, dstDesc, gmCachePolicy);
    Catlass::Epilogue::Tile::CopyUb2GmTla<ArchTag, decltype(srcTensor), decltype(dstTensor)>{}(dstTensor, srcTensor);
}

// UB -> UB. DataCopy with DataCopyParams lowers to MOV.UB.TO.UB (arch 3510), a
// single block move that carries any element width and any row length.
//
// The move is block-based: blockLen and both gaps are counts of 32-byte blocks.
// A row whose byte length is not a whole number of blocks is rounded up, so the
// move carries the row's own 32-byte alignment fill along with the row, and the
// gaps are measured from the rounded row end. tla.copy admits only the
// geometries where that is a copy -- both origins on a block boundary, and a row
// pitch with whole-block slack for the rounded row -- which is what makes the
// block arithmetic below exact. The guard is
// _ubub_require_metaop_representable in core_api.py.
template <class ArchTag, LayoutTag SrcLayout, LayoutTag DstLayout, typename T>
CATLASS_DEVICE void copyUBToUB(
    memref_t<__ubuf__ T, 1>* src, memref_t<__ubuf__ T, 1>* dst, const TensorDesc& srcDesc, const TensorDesc& dstDesc)
{
    AscendC::LocalTensor<T> srcTensor(AscendC::TPosition::VECCALC, localAddr(src), elementCount(src));
    AscendC::LocalTensor<T> dstTensor(AscendC::TPosition::VECCALC, localAddr(dst), elementCount(dst));

    constexpr uint32_t kBlockBytes = 32;
    // originShape is the tile's valid extent; shape is the packing it is
    // addressed through, which a tail tile pads out beyond that extent. The move
    // carries the valid elements, so it is sized from the origin, as every other
    // route here is.
    const uint32_t rows = static_cast<uint32_t>(srcDesc.originShape0);
    const uint32_t cols = static_cast<uint32_t>(srcDesc.originShape1);
    const uint32_t rowBytes = cols * static_cast<uint32_t>(sizeof(T));
    const uint32_t srcPitchBytes = static_cast<uint32_t>(srcDesc.stride0) * static_cast<uint32_t>(sizeof(T));
    const uint32_t dstPitchBytes = static_cast<uint32_t>(dstDesc.stride0) * static_cast<uint32_t>(sizeof(T));

    // The descriptor's coordinate is the tile's origin inside its parent, in
    // elements. Every other route reaches it through makeTlaTileCoord(), which
    // builds a tla::Tensor that folds the coordinate into the address; this one
    // drives DataCopy from a raw LocalTensor, so it has to apply the coordinate
    // itself. Without this a sub-block view (a padded 64x32 slice of a 64x64
    // tile, say) copies at the parent's origin instead, silently landing the
    // wrong data -- the row pitch below is honoured, but the base is not.
    const uint32_t srcElemOffset =
        static_cast<uint32_t>(static_cast<int64_t>(srcDesc.coord0) * srcDesc.stride0 + srcDesc.coord1);
    const uint32_t dstElemOffset =
        static_cast<uint32_t>(static_cast<int64_t>(dstDesc.coord0) * dstDesc.stride0 + dstDesc.coord1);

    // Rounding the row length up is what makes every row length expressible.
    // The alternative is to round down, which is what AscendC's own Level-2
    // overload does -- its header states that no error is reported in NPU mode
    // and that the value is rounded down by 32B -- and that drops the tail of
    // every row. There is no DataCopyPad for a local-to-local move to pick
    // instead, so rounding up, into the row's own alignment fill, is the only
    // direction that moves all of the bytes.
    const uint32_t rowBlocks = (rowBytes + kBlockBytes - 1) / kBlockBytes;
    const uint32_t rowBlockBytes = rowBlocks * kBlockBytes;
    // A single-row copy has no gap to express, and its pitch is allowed to be
    // narrower than the rounded row, so clamp rather than underflow.
    const uint32_t srcGapBlocks = srcPitchBytes > rowBlockBytes ? (srcPitchBytes - rowBlockBytes) / kBlockBytes : 0;
    const uint32_t dstGapBlocks = dstPitchBytes > rowBlockBytes ? (dstPitchBytes - rowBlockBytes) / kBlockBytes : 0;

    AscendC::DataCopyParams params;
    params.blockCount = static_cast<uint16_t>(rows);
    params.blockLen = static_cast<uint16_t>(rowBlocks);
    params.srcGap = static_cast<uint16_t>(srcGapBlocks);
    params.dstGap = static_cast<uint16_t>(dstGapBlocks);
    AscendC::DataCopy(dstTensor[dstElemOffset], srcTensor[srcElemOffset], params);
}

// UB -> UB, transposing: dst[j][i] = src[i][j].
//
// src is a logical (M, N) RowMajor tile whose rows are srcDesc.stride0 elements
// apart. dst carries the same logical (M, N) shape tagged ColumnMajor, which is
// physically an (N, M) RowMajor array whose rows are dstDesc.stride1 elements
// apart -- so the copy is a transpose in memory with no shape bookkeeping on the
// caller's side.
//
// The engine is VNCHWCONV (AscendC::TransDataTo5HD), the vector unit's native
// transpose. It takes 16 source and 16 destination addresses through the VA
// registers and transposes the block they span. A 32-byte DataBlock holds
// C0 = 32 / sizeof(T) elements, so the natural block is 16 rows x C0 columns:
// square (16x16) for b16, and 16x8 / 16x32 for b32 / b8, where the wider side is
// covered by pairing up the address list.
template <typename T>
CATLASS_DEVICE void transposeTileUB(
    __ubuf__ T* dst, __ubuf__ T* src, uint32_t rows, uint32_t cols, uint32_t srcPitch, uint32_t dstPitch)
{
    constexpr uint32_t kList = 16;           // VA address-list length
    constexpr uint32_t kC0 = 32 / sizeof(T); // elements in a 32-byte DataBlock
    constexpr uint32_t kMaxRepeat = 255;
    // b8 needs four calls to cover a tile, and they span 32 source rows; every
    // other width covers its tile from one address list of 16 rows.
    constexpr uint32_t kTileRows = (sizeof(T) == 1) ? 2 * kList : kList;

    // Walking one column block to the next moves src by kC0 elements -- exactly
    // one 32-byte DataBlock for every width -- and dst by kC0 transposed rows,
    // i.e. kC0 * dstPitch elements, which is dstPitch DataBlocks. Both steps are
    // uniform, so the whole row of blocks is one call with repeatTimes and the 16
    // MOVEVA that load the address list are paid once per row block instead of
    // once per block. (CANN's own 3510 cumsum transpose uses these same two
    // strides.)
    const uint16_t srcRepStride = 1;
    const uint16_t dstRepStride = static_cast<uint16_t>(dstPitch);

    uint64_t srcList[kList];
    uint64_t dstList[kList];

    for (uint32_t i0 = 0; i0 < rows; i0 += kTileRows) {
        for (uint32_t j0 = 0; j0 < cols; j0 += kC0 * kMaxRepeat) {
            const uint32_t left = (cols - j0) / kC0;
            const uint32_t repeats = (left < kMaxRepeat) ? left : kMaxRepeat;
            AscendC::TransDataTo5HDParams params(
                false, false, static_cast<uint8_t>(repeats), repeats == 1 ? 0 : dstRepStride,
                repeats == 1 ? 0 : srcRepStride);

            if constexpr (sizeof(T) == 2) {
                // The b16 block is square: 16 source rows in, 16 transposed rows
                // out, one destination address each.
                for (uint32_t k = 0; k < kList; ++k) {
                    srcList[k] = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(src + (i0 + k) * srcPitch + j0));
                    dstList[k] = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(dst + (j0 + k) * dstPitch + i0));
                }
                AscendC::TransDataTo5HD<T>(dstList, srcList, params);
            } else if constexpr (sizeof(T) == 4) {
                // The b32 block is 16 rows x 8 columns, so it transposes to 8 rows
                // of 16 elements -- two 32-byte destination blocks per row, which
                // is why the destination list is filled in pairs.
                for (uint32_t k = 0; k < kList; ++k) {
                    srcList[k] = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(src + (i0 + k) * srcPitch + j0));
                }
                for (uint32_t k = 0; k < kC0; ++k) {
                    __ubuf__ T* row = dst + (j0 + k) * dstPitch + i0;
                    dstList[2 * k] = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(row));
                    dstList[2 * k + 1] = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(row + kC0));
                }
                AscendC::TransDataTo5HD<T>(dstList, srcList, params);
            } else {
                // The b8 DataBlock holds 32 elements but the address list is only
                // 16 entries long, so one call moves a 16x16 quarter of the 32x32
                // tile: srcHighHalf picks which 16 of the block's 32 columns are
                // read, dstHighHalf which 16-byte half of the destination block is
                // written (equivalently, which 16 source rows feed it).
                for (uint32_t q = 0; q < 4; ++q) {
                    const bool srcHigh = (q & 1) != 0; // columns j0+16..j0+31
                    const bool dstHigh = (q & 2) != 0; // source rows i0+16..i0+31
                    const uint32_t rowBase = i0 + (dstHigh ? kList : 0);
                    const uint32_t colBase = j0 + (srcHigh ? kList : 0);
                    for (uint32_t k = 0; k < kList; ++k) {
                        srcList[k] =
                            static_cast<uint64_t>(reinterpret_cast<uintptr_t>(src + (rowBase + k) * srcPitch + j0));
                        dstList[k] =
                            static_cast<uint64_t>(reinterpret_cast<uintptr_t>(dst + (colBase + k) * dstPitch + i0));
                    }
                    params.srcHighHalf = srcHigh;
                    params.dstHighHalf = dstHigh;
                    AscendC::TransDataTo5HD<T>(dstList, srcList, params);
                }
            }
        }
    }
}

template <class ArchTag, LayoutTag SrcLayout, LayoutTag DstLayout, typename T>
CATLASS_DEVICE void copyUBToUBTranspose(
    memref_t<__ubuf__ T, 1>* src, memref_t<__ubuf__ T, 1>* dst, const TensorDesc& srcDesc, const TensorDesc& dstDesc)
{
    __ubuf__ T* srcPtr = reinterpret_cast<__ubuf__ T*>(static_cast<uintptr_t>(localAddr(src)));
    __ubuf__ T* dstPtr = reinterpret_cast<__ubuf__ T*>(static_cast<uintptr_t>(localAddr(dst)));
    // The descriptor's coordinate is the tile's origin inside its parent, in
    // elements, and localAddr() carries only the allocation's own base. Like the
    // straight UB -> UB move above, this route drives the hardware from a raw
    // pointer rather than from a tla::Tensor, so nothing folds the coordinate in
    // for it and it has to do so itself.
    //
    // The two sides fold differently, because the destination is ColumnMajor:
    // src is a RowMajor (M, N) tile pitched by stride0, so its origin is at
    // coord0 rows down and coord1 elements across; dst carries the same logical
    // (M, N) coordinate over an (N, M) array pitched by stride1, so there the
    // logical column is the physical row and the two terms swap. Folding the
    // destination the way the source folds would land every non-square offset
    // tile at the wrong address.
    srcPtr += static_cast<uint32_t>(static_cast<int64_t>(srcDesc.coord0) * srcDesc.stride0 + srcDesc.coord1);
    dstPtr += static_cast<uint32_t>(static_cast<int64_t>(dstDesc.coord1) * dstDesc.stride1 + dstDesc.coord0);
    // originShape is the tile's valid extent; shape is the packing it is
    // addressed through, which a tail tile pads out beyond that extent. The
    // transpose carries the valid elements, so it is sized from the origin, as
    // the straight UB -> UB move and every other route here are.
    transposeTileUB<T>(
        dstPtr, srcPtr, static_cast<uint32_t>(srcDesc.originShape0), static_cast<uint32_t>(srcDesc.originShape1),
        static_cast<uint32_t>(srcDesc.stride0), static_cast<uint32_t>(dstDesc.stride1));
}

template <class ArchTag, LayoutTag SrcLayout, LayoutTag DstLayout, typename T>
CATLASS_DEVICE void copyUBToL1(
    memref_t<__ubuf__ T, 1>* src, memref_t<__cbuf__ T, 1>* dst, const TensorDesc& srcDesc, const TensorDesc& dstDesc)
{
    auto srcTensor = makeUBTensor<SrcLayout, T>(src, srcDesc);
    auto dstTensor = makeL1Tensor<DstLayout, T>(dst, dstDesc);
    Catlass::Epilogue::Tile::CopyUb2L1Tla<ArchTag, decltype(srcTensor), decltype(dstTensor)>{}(dstTensor, srcTensor);
}

// fp8 on the vector routes, staged as bytes.
//
// The AIV backend refuses fp8 outright -- "type only supports pointer
// operations, scalar float type semantics are not supported" -- and the UB
// copy templates trip that, though the cube's DMA-only ones do not. Pointer
// operations *are* allowed, so the symbol keeps the fp8 name and signature the
// route resolves to, and only the copy template is instantiated on int8_t. A
// copy does not interpret the element type beyond stride and layout
// arithmetic, and fp8 is a byte, so the generated move is the same one.
//
// This is the byte-staging workaround that every fp8 kernel used to carry by
// hand, moved down to where it belongs: the caller now sees a typed fp8 tile.
#define REGISTER_GM_TO_UB_AS_BYTES(LayoutSrc, LayoutDst, DType)                                                       \
    [aicore] __attribute__((always_inline)) void _mlir_ciface_copy_gm_##LayoutSrc##_to_ub_##LayoutDst##_##DType(      \
        memref_t<__gm__ DType, 2>* src, memref_t<__ubuf__ DType, 1>* dst, DESC_ABI_PARAMS(src), DESC_ABI_PARAMS(dst), \
        int64_t gmCachePolicy)                                                                                        \
    {                                                                                                                 \
        copyGMToUB<Catlass::Arch::Ascend950, LayoutTag::LayoutSrc, LayoutTag::LayoutDst, int8_t>(                     \
            reinterpret_cast<memref_t<__gm__ int8_t, 2>*>(src), reinterpret_cast<memref_t<__ubuf__ int8_t, 1>*>(dst), \
            TENSOR_DESC_12(src), TENSOR_DESC_12(dst), gmCachePolicy);                                                 \
    }

#define REGISTER_UB_TO_GM_AS_BYTES(LayoutSrc, LayoutDst, DType)                                                       \
    [aicore] __attribute__((always_inline)) void _mlir_ciface_copy_ub_##LayoutSrc##_to_gm_##LayoutDst##_##DType(      \
        memref_t<__ubuf__ DType, 1>* src, memref_t<__gm__ DType, 2>* dst, DESC_ABI_PARAMS(src), DESC_ABI_PARAMS(dst), \
        int64_t gmCachePolicy)                                                                                        \
    {                                                                                                                 \
        copyUBToGM<Catlass::Arch::Ascend950, LayoutTag::LayoutSrc, LayoutTag::LayoutDst, int8_t>(                     \
            reinterpret_cast<memref_t<__ubuf__ int8_t, 1>*>(src), reinterpret_cast<memref_t<__gm__ int8_t, 2>*>(dst), \
            TENSOR_DESC_12(src), TENSOR_DESC_12(dst), gmCachePolicy);                                                 \
    }

#define REGISTER_UB_TO_L1_AS_BYTES(LayoutSrc, LayoutDst, DType)                                                  \
    [aicore] __attribute__((always_inline)) void _mlir_ciface_copy_ub_##LayoutSrc##_to_l1_##LayoutDst##_##DType( \
        memref_t<__ubuf__ DType, 1>* src, memref_t<__cbuf__ DType, 1>* dst, DESC_ABI_PARAMS(src),                \
        DESC_ABI_PARAMS(dst))                                                                                    \
    {                                                                                                            \
        copyUBToL1<Catlass::Arch::Ascend950, LayoutTag::LayoutSrc, LayoutTag::LayoutDst, int8_t>(                \
            reinterpret_cast<memref_t<__ubuf__ int8_t, 1>*>(src),                                                \
            reinterpret_cast<memref_t<__cbuf__ int8_t, 1>*>(dst), TENSOR_DESC_12(src), TENSOR_DESC_12(dst));     \
    }

// Packed fp4 on the vector routes.
//
// fp4 cannot reach the copy templates at all: the AIV backend refuses every
// sub-byte float outright -- "fp4e2m1x2 ... type only supports pointer
// operations, scalar float type semantics are not supported" -- the same wall
// fp8 hit. Unlike fp8 the workaround is not a plain reinterpret, because two
// fp4 elements share a byte, so the descriptor is converted to byte
// denomination here and the copy runs on int8_t.
//
// Only the packed axis changes: a RowMajor tile of (rows, cols) fp4 elements is
// (rows, cols/2) bytes with half the row stride, so the row count and any
// coordinate along it are untouched.
//
// Every division below is exact. The routes that convert a descriptor here --
// GM <-> UB, and UB -> L1 through copyUBToL1Fp4 below -- accept only tiles
// whose packed extent, packed origin coordinate and halved stride are all
// even. An odd one of the three would end mid-byte, putting the other half of
// that byte -- an element outside the tile -- in the way, and the copy could
// neither carry it nor disturb it. core_api.py's _partial_byte_fp4_extent and
// _fp4_odd_byte_mapping refuse those shapes at trace time, including values
// that are only known at run time, and carry the full reasoning.
//
// Which stride is the halved one is per layout, so the frontend checks it per
// layout: stride0 on a RowMajor tile, stride3 on a zN one. Both are reachable
// with an odd value only through an explicit layout, since every stride the
// frontend materializes for fp4 is already even.
CATLASS_DEVICE TensorDesc fp4RowMajorDescToBytes(const TensorDesc& d)
{
    TensorDesc b = d;
    b.shape1 = d.shape1 / 2;
    b.stride0 = d.stride0 / 2;
    b.originShape1 = d.originShape1 / 2;
    b.coord1 = d.coord1 / 2;
    return b;
}

// The zN counterpart of fp4RowMajorDescToBytes. zN partitions into 32-byte C0s
// whatever the element width, so an fp4 zN tile and an int8 zN tile of half the
// columns are the same bytes: the fractal shape is untouched and only the
// column extent and the stride between C0 columns halve.
//
// stride3 is that stride between C0 columns, and halving it is exact because
// the frontend requires it to be even. A layout the frontend materializes
// derives it as a whole number of C0s, but an explicit layout can carry any
// value, so an odd one is refused at trace time rather than assumed away here.
CATLASS_DEVICE TensorDesc fp4zNDescToBytes(const TensorDesc& d)
{
    TensorDesc b = d;
    b.stride3 = d.stride3 / 2;
    b.originShape1 = d.originShape1 / 2;
    b.coord1 = d.coord1 / 2;
    return b;
}

// UB -> L1 goes through int8 tensors and byte descriptors, exactly as the
// GM <-> UB fp4 routes do, and carries the same whole-byte precondition: a
// RowMajor source converts through fp4RowMajorDescToBytes and a zN tile
// through fp4zNDescToBytes, and both halve the packed extent and the packed
// origin coordinate.
//
// The destination is always zN and always converts through fp4zNDescToBytes,
// so its C0-column pitch carries the even requirement on both registered
// pairings -- (RowMajor, zN) as well as (zN, zN) -- and not only where the
// source is zN.
//
// Keeping the real fp4 type here instead -- on the argument that the layout
// math runs on SizeOfBits and no scalar float value is ever formed -- compiles,
// and then faults on device (error type 3, 507015) for every shape tried, down
// to one block with one K tile and a single DataCopy. The GM <-> UB round trip
// passes over the same buffers, and skipping only this store lets the same
// kernel run clean, so the typed instantiation is what does not work.
template <class ArchTag, typename TFp4, LayoutTag SrcLayout, LayoutTag DstLayout>
CATLASS_DEVICE void copyUBToL1Fp4(
    memref_t<__ubuf__ int8_t, 1>* src, memref_t<__cbuf__ int8_t, 1>* dst, const TensorDesc& srcDesc,
    const TensorDesc& dstDesc)
{
    auto srcBytes = (SrcLayout == LayoutTag::RowMajor) ? fp4RowMajorDescToBytes(srcDesc) : fp4zNDescToBytes(srcDesc);
    auto srcTensor = makeUBTensor<SrcLayout, int8_t>(src, srcBytes);
    auto dstTensor = makeL1Tensor<DstLayout, int8_t>(dst, fp4zNDescToBytes(dstDesc));
    Catlass::Epilogue::Tile::CopyUb2L1Tla<ArchTag, decltype(srcTensor), decltype(dstTensor)>{}(dstTensor, srcTensor);
}

// The MX block scale, UB -> L1. This is what lets a scale computed on the
// vector unit reach the cube without a detour through global memory: the
// element tile takes the zN route above and the scale takes this one, and the
// two are joined at the L1 -> L0 load, which is where MX has always joined them.
template <class ArchTag, LayoutTag SrcLayout, LayoutTag DstLayout>
CATLASS_DEVICE void copyUBToL1MxScale(
    memref_t<__ubuf__ uint8_t, 1>* src, memref_t<__cbuf__ uint8_t, 1>* dst, const TensorDesc& srcDesc,
    const TensorDesc& dstDesc)
{
    auto srcTensor = makeUBMxScaleTensor<SrcLayout>(src, srcDesc);
    auto dstTensor = makeL1MxScaleTensor<DstLayout>(dst, dstDesc);
    Catlass::Epilogue::Tile::CopyUb2L1Tla<ArchTag, decltype(srcTensor), decltype(dstTensor)>{}(dstTensor, srcTensor);
}

extern "C" {
#if ((defined(__NPU_ARCH__) && __NPU_ARCH__ == 3510) || (defined(CATLASS_ARCH) && CATLASS_ARCH == 3510))

#define REGISTER_GM_TO_UB(LayoutSrc, LayoutDst, DType)                                                                \
    [aicore] __attribute__((always_inline)) void _mlir_ciface_copy_gm_##LayoutSrc##_to_ub_##LayoutDst##_##DType(      \
        memref_t<__gm__ DType, 2>* src, memref_t<__ubuf__ DType, 1>* dst, DESC_ABI_PARAMS(src), DESC_ABI_PARAMS(dst), \
        int64_t gmCachePolicy)                                                                                        \
    {                                                                                                                 \
        copyGMToUB<Catlass::Arch::Ascend950, LayoutTag::LayoutSrc, LayoutTag::LayoutDst, DType>(                      \
            src, dst, TENSOR_DESC_12(src), TENSOR_DESC_12(dst), gmCachePolicy);                                       \
    }

#define REGISTER_UB_TO_GM(LayoutSrc, LayoutDst, DType)                                                                \
    [aicore] __attribute__((always_inline)) void _mlir_ciface_copy_ub_##LayoutSrc##_to_gm_##LayoutDst##_##DType(      \
        memref_t<__ubuf__ DType, 1>* src, memref_t<__gm__ DType, 2>* dst, DESC_ABI_PARAMS(src), DESC_ABI_PARAMS(dst), \
        int64_t gmCachePolicy)                                                                                        \
    {                                                                                                                 \
        copyUBToGM<Catlass::Arch::Ascend950, LayoutTag::LayoutSrc, LayoutTag::LayoutDst, DType>(                      \
            src, dst, TENSOR_DESC_12(src), TENSOR_DESC_12(dst), gmCachePolicy);                                       \
    }

#define REGISTER_UB_TO_UB(LayoutSrc, LayoutDst, DType)                                                           \
    [aicore] __attribute__((always_inline)) void _mlir_ciface_copy_ub_##LayoutSrc##_to_ub_##LayoutDst##_##DType( \
        memref_t<__ubuf__ DType, 1>* src, memref_t<__ubuf__ DType, 1>* dst, DESC_ABI_PARAMS(src),                \
        DESC_ABI_PARAMS(dst))                                                                                    \
    {                                                                                                            \
        copyUBToUB<Catlass::Arch::Ascend950, LayoutTag::LayoutSrc, LayoutTag::LayoutDst, DType>(                 \
            src, dst, TENSOR_DESC_12(src), TENSOR_DESC_12(dst));                                                 \
    }

#define REGISTER_UB_TO_L1(LayoutSrc, LayoutDst, DType)                                                           \
    [aicore] __attribute__((always_inline)) void _mlir_ciface_copy_ub_##LayoutSrc##_to_l1_##LayoutDst##_##DType( \
        memref_t<__ubuf__ DType, 1>* src, memref_t<__cbuf__ DType, 1>* dst, DESC_ABI_PARAMS(src),                \
        DESC_ABI_PARAMS(dst))                                                                                    \
    {                                                                                                            \
        copyUBToL1<Catlass::Arch::Ascend950, LayoutTag::LayoutSrc, LayoutTag::LayoutDst, DType>(                 \
            src, dst, TENSOR_DESC_12(src), TENSOR_DESC_12(dst));                                                 \
    }

REGISTER_GM_TO_UB(RowMajor, RowMajor, float)
REGISTER_GM_TO_UB(RowMajor, RowMajor, half)
REGISTER_GM_TO_UB(RowMajor, RowMajor, bf16)
REGISTER_GM_TO_UB(RowMajor, RowMajor, int32_t)
REGISTER_GM_TO_UB(RowMajor, RowMajor, int16_t)
REGISTER_GM_TO_UB(RowMajor, RowMajor, int8_t)
// Unsigned staging. A copy does not interpret the element type beyond stride
// and layout arithmetic, so these are the signed moves at the same widths; they
// exist as their own symbols because the route resolves by element type.
// Without them an unsigned tile cannot reach UB for Compare/Cast. u64 is
// omitted deliberately: i64 has no GM<->UB route either.
REGISTER_GM_TO_UB(RowMajor, RowMajor, uint8_t)
REGISTER_GM_TO_UB(RowMajor, RowMajor, uint16_t)
REGISTER_GM_TO_UB(RowMajor, RowMajor, uint32_t)
REGISTER_GM_TO_UB_AS_BYTES(RowMajor, RowMajor, fp8_e4m3fn_t)
REGISTER_GM_TO_UB_AS_BYTES(RowMajor, RowMajor, fp8_e5m2_t)

REGISTER_UB_TO_GM(RowMajor, RowMajor, float)
REGISTER_UB_TO_GM(RowMajor, RowMajor, half)
REGISTER_UB_TO_GM(RowMajor, RowMajor, bf16)
REGISTER_UB_TO_GM(RowMajor, RowMajor, int32_t)
REGISTER_UB_TO_GM(RowMajor, RowMajor, int16_t)
REGISTER_UB_TO_GM(RowMajor, RowMajor, int8_t)
REGISTER_UB_TO_GM(RowMajor, RowMajor, uint8_t)
REGISTER_UB_TO_GM(RowMajor, RowMajor, uint16_t)
REGISTER_UB_TO_GM(RowMajor, RowMajor, uint32_t)
REGISTER_UB_TO_GM_AS_BYTES(RowMajor, RowMajor, fp8_e4m3fn_t)
REGISTER_UB_TO_GM_AS_BYTES(RowMajor, RowMajor, fp8_e5m2_t)
// The MX block scale on the vector routes. An e8m0 exponent is one opaque byte,
// so it stages exactly like fp8 -- the AIV refuses the type in the copy
// templates for the same reason, and the same byte instantiation carries it.
// This is what lets a kernel compute scales on the vector unit and get them
// back out; the reordering into an L1 MxScale block is a separate leg.
REGISTER_GM_TO_UB_AS_BYTES(RowMajor, RowMajor, float8_e8m0_t)
REGISTER_UB_TO_GM_AS_BYTES(RowMajor, RowMajor, float8_e8m0_t)

#define REGISTER_UB_TO_L1_MX_SCALE(SrcLayout, DstLayout)                                                         \
    [aicore] __attribute__((always_inline)) void _mlir_ciface_copy_ub_##SrcLayout##_to_l1_##DstLayout##_uint8_t( \
        memref_t<__ubuf__ uint8_t, 1>* src, memref_t<__cbuf__ uint8_t, 1>* dst, DESC_ABI_PARAMS(src),            \
        DESC_ABI_PARAMS(dst))                                                                                    \
    {                                                                                                            \
        copyUBToL1MxScale<Catlass::Arch::Ascend950, LayoutTag::SrcLayout, LayoutTag::DstLayout>(                 \
            src, dst, TENSOR_DESC_12(src), TENSOR_DESC_12(dst));                                                 \
    }

REGISTER_UB_TO_L1_MX_SCALE(zZMxScale, zZMxScale)
REGISTER_UB_TO_L1_MX_SCALE(nNMxScale, nNMxScale)

// fp8 rides the route as bytes: DataCopy moves blocks, so the element type only
// has to describe the width. int8_t is registered directly.
#define REGISTER_UB_TO_UB_AS_BYTES(LayoutSrc, LayoutDst, DType)                                                  \
    [aicore] __attribute__((always_inline)) void _mlir_ciface_copy_ub_##LayoutSrc##_to_ub_##LayoutDst##_##DType( \
        memref_t<__ubuf__ DType, 1>* src, memref_t<__ubuf__ DType, 1>* dst, DESC_ABI_PARAMS(src),                \
        DESC_ABI_PARAMS(dst))                                                                                    \
    {                                                                                                            \
        copyUBToUB<Catlass::Arch::Ascend950, LayoutTag::LayoutSrc, LayoutTag::LayoutDst, int8_t>(                \
            reinterpret_cast<memref_t<__ubuf__ int8_t, 1>*>(src),                                                \
            reinterpret_cast<memref_t<__ubuf__ int8_t, 1>*>(dst), TENSOR_DESC_12(src), TENSOR_DESC_12(dst));     \
    }

REGISTER_UB_TO_UB(RowMajor, RowMajor, float)
REGISTER_UB_TO_UB(RowMajor, RowMajor, half)
REGISTER_UB_TO_UB(RowMajor, RowMajor, bf16)
REGISTER_UB_TO_UB(RowMajor, RowMajor, int32_t)
REGISTER_UB_TO_UB(RowMajor, RowMajor, int16_t)
REGISTER_UB_TO_UB(RowMajor, RowMajor, int8_t)
REGISTER_UB_TO_UB_AS_BYTES(RowMajor, RowMajor, fp8_e4m3fn_t)
REGISTER_UB_TO_UB_AS_BYTES(RowMajor, RowMajor, fp8_e5m2_t)

// Transposing UB -> UB: the destination tile is tagged ColumnMajor, so the same
// logical (M, N) shape lands as an (N, M) array. VNCHWCONV carries b8/b16/b32,
// and fp8 rides as bytes exactly as the straight copy does.
#define REGISTER_UB_TO_UB_T(LayoutSrc, LayoutDst, DType)                                                         \
    [aicore] __attribute__((always_inline)) void _mlir_ciface_copy_ub_##LayoutSrc##_to_ub_##LayoutDst##_##DType( \
        memref_t<__ubuf__ DType, 1>* src, memref_t<__ubuf__ DType, 1>* dst, DESC_ABI_PARAMS(src),                \
        DESC_ABI_PARAMS(dst))                                                                                    \
    {                                                                                                            \
        copyUBToUBTranspose<Catlass::Arch::Ascend950, LayoutTag::LayoutSrc, LayoutTag::LayoutDst, DType>(        \
            src, dst, TENSOR_DESC_12(src), TENSOR_DESC_12(dst));                                                 \
    }

#define REGISTER_UB_TO_UB_T_AS_BYTES(LayoutSrc, LayoutDst, DType)                                                \
    [aicore] __attribute__((always_inline)) void _mlir_ciface_copy_ub_##LayoutSrc##_to_ub_##LayoutDst##_##DType( \
        memref_t<__ubuf__ DType, 1>* src, memref_t<__ubuf__ DType, 1>* dst, DESC_ABI_PARAMS(src),                \
        DESC_ABI_PARAMS(dst))                                                                                    \
    {                                                                                                            \
        copyUBToUBTranspose<Catlass::Arch::Ascend950, LayoutTag::LayoutSrc, LayoutTag::LayoutDst, int8_t>(       \
            reinterpret_cast<memref_t<__ubuf__ int8_t, 1>*>(src),                                                \
            reinterpret_cast<memref_t<__ubuf__ int8_t, 1>*>(dst), TENSOR_DESC_12(src), TENSOR_DESC_12(dst));     \
    }

REGISTER_UB_TO_UB_T(RowMajor, ColumnMajor, float)
REGISTER_UB_TO_UB_T(RowMajor, ColumnMajor, half)
REGISTER_UB_TO_UB_T(RowMajor, ColumnMajor, bf16)
REGISTER_UB_TO_UB_T(RowMajor, ColumnMajor, int32_t)
REGISTER_UB_TO_UB_T(RowMajor, ColumnMajor, int16_t)
REGISTER_UB_TO_UB_T(RowMajor, ColumnMajor, int8_t)
REGISTER_UB_TO_UB_T_AS_BYTES(RowMajor, ColumnMajor, fp8_e4m3fn_t)
REGISTER_UB_TO_UB_T_AS_BYTES(RowMajor, ColumnMajor, fp8_e5m2_t)

REGISTER_UB_TO_L1(RowMajor, zN, float)
REGISTER_UB_TO_L1(RowMajor, zN, half)
REGISTER_UB_TO_L1(RowMajor, zN, bf16)
// Byte staging, for data the vector core carries opaquely: a sub-byte format
// such as fp4 cannot be named on any route, so a kernel stages it as raw bytes
// and re-types it with tla.recast_ptr once it is in L1. fp8 does not need this
// -- it has its own typed UB->L1 entries below. GM<->UB already register
// int8_t; without the matching UB->L1 the staging path stops one hop short of
// the cube.
REGISTER_UB_TO_L1(RowMajor, zN, int8_t)
REGISTER_UB_TO_L1(zN, zN, float)
REGISTER_UB_TO_L1(zN, zN, half)
REGISTER_UB_TO_L1(zN, zN, bf16)
REGISTER_UB_TO_L1(zN, zN, int8_t)
REGISTER_UB_TO_L1_AS_BYTES(RowMajor, zN, fp8_e4m3fn_t)
REGISTER_UB_TO_L1_AS_BYTES(RowMajor, zN, fp8_e5m2_t)
REGISTER_UB_TO_L1_AS_BYTES(zN, zN, fp8_e4m3fn_t)
REGISTER_UB_TO_L1_AS_BYTES(zN, zN, fp8_e5m2_t)

// Named by the fp4 encoding rather than by a storage suffix, the same way the
// cube's GM -> L1 fp4 routes are: no byte type would say which of the two 4-bit
// formats the tile holds, and the caller's tile is typed fp4 on both sides.
#define REGISTER_GM_TO_UB_FP4(NameSrc, NameDst, TFp4)                                                           \
    [aicore] __attribute__((always_inline)) void _mlir_ciface_copy_gm_##NameSrc##_to_ub_##NameDst##_##TFp4(     \
        memref_t<__gm__ int8_t, 2>* src, memref_t<__ubuf__ int8_t, 1>* dst, DESC_ABI_PARAMS(src),               \
        DESC_ABI_PARAMS(dst), int64_t gmCachePolicy)                                                            \
    {                                                                                                           \
        copyGMToUB<Catlass::Arch::Ascend950, LayoutTag::NameSrc, LayoutTag::NameDst, int8_t>(                   \
            src, dst, fp4RowMajorDescToBytes(TENSOR_DESC_12(src)), fp4RowMajorDescToBytes(TENSOR_DESC_12(dst)), \
            gmCachePolicy);                                                                                     \
    }

#define REGISTER_UB_TO_GM_FP4(NameSrc, NameDst, TFp4)                                                           \
    [aicore] __attribute__((always_inline)) void _mlir_ciface_copy_ub_##NameSrc##_to_gm_##NameDst##_##TFp4(     \
        memref_t<__ubuf__ int8_t, 1>* src, memref_t<__gm__ int8_t, 2>* dst, DESC_ABI_PARAMS(src),               \
        DESC_ABI_PARAMS(dst), int64_t gmCachePolicy)                                                            \
    {                                                                                                           \
        copyUBToGM<Catlass::Arch::Ascend950, LayoutTag::NameSrc, LayoutTag::NameDst, int8_t>(                   \
            src, dst, fp4RowMajorDescToBytes(TENSOR_DESC_12(src)), fp4RowMajorDescToBytes(TENSOR_DESC_12(dst)), \
            gmCachePolicy);                                                                                     \
    }

REGISTER_GM_TO_UB_FP4(RowMajor, RowMajor, float4_e2m1x2_t)
REGISTER_GM_TO_UB_FP4(RowMajor, RowMajor, float4_e1m2x2_t)
REGISTER_UB_TO_GM_FP4(RowMajor, RowMajor, float4_e2m1x2_t)
REGISTER_UB_TO_GM_FP4(RowMajor, RowMajor, float4_e1m2x2_t)

#define REGISTER_UB_TO_L1_FP4(NameSrc, NameDst, TFp4)                                                       \
    [aicore] __attribute__((always_inline)) void _mlir_ciface_copy_ub_##NameSrc##_to_l1_##NameDst##_##TFp4( \
        memref_t<__ubuf__ int8_t, 1>* src, memref_t<__cbuf__ int8_t, 1>* dst, DESC_ABI_PARAMS(src),         \
        DESC_ABI_PARAMS(dst))                                                                               \
    {                                                                                                       \
        copyUBToL1Fp4<Catlass::Arch::Ascend950, TFp4, LayoutTag::NameSrc, LayoutTag::NameDst>(              \
            src, dst, TENSOR_DESC_12(src), TENSOR_DESC_12(dst));                                            \
    }

REGISTER_UB_TO_L1_FP4(RowMajor, zN, float4_e2m1x2_t)
REGISTER_UB_TO_L1_FP4(RowMajor, zN, float4_e1m2x2_t)
REGISTER_UB_TO_L1_FP4(zN, zN, float4_e2m1x2_t)
REGISTER_UB_TO_L1_FP4(zN, zN, float4_e1m2x2_t)

#endif
}
