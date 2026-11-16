#include <algorithm>
#include <cstddef>
using std::size_t;

#include <kernel_operator.h>

#include "catlass/arch/arch.hpp"
#include "catlass/catlass.hpp"
#include "catlass/gemm/block/block_mmad.hpp"
#include "catlass/gemm/block/block_swizzle.hpp"
#include "catlass/gemm/dispatch_policy.hpp"
#include "catlass/gemm/gemm_type.hpp"
#include "catlass/gemm/kernel/batched_matmul_fixpipe_dequant_tla.hpp"
#include "catlass/layout/layout.hpp"
#include "tla/layout.hpp"

#include "catlass_kernel.h"
#include "common/kernel_runner.h"
#include "common/tile_shape_scaler_tla.h"

#ifndef CATLASS_JIT_ELEMENT_A
#define CATLASS_JIT_ELEMENT_A hifloat8_t
#endif
#ifndef CATLASS_JIT_ELEMENT_B
#define CATLASS_JIT_ELEMENT_B hifloat8_t
#endif
// The kernel output is always BF16.
#define CATLASS_JIT_ELEMENT_C bfloat16_t
#ifndef CATLASS_JIT_LAYOUT_A
#define CATLASS_JIT_LAYOUT_A RowMajor
#endif
#ifndef CATLASS_JIT_LAYOUT_B
#define CATLASS_JIT_LAYOUT_B RowMajor
#endif

using namespace Catlass;

using ElementA = CATLASS_JIT_ELEMENT_A;
using ElementB = CATLASS_JIT_ELEMENT_B;
using ElementC = CATLASS_JIT_ELEMENT_C;

using LayoutTagA = layout::CATLASS_JIT_LAYOUT_A;
using LayoutTagB = layout::CATLASS_JIT_LAYOUT_B;
using LayoutTagC = layout::RowMajor;

using ElementBias = float;

using ArchTag = Arch::Ascend950;
constexpr bool enableUnitFlag = true;
constexpr bool useHF32 = false;
constexpr bool enableL1Resident = false;
constexpr uint32_t l0CStages = 1;
constexpr uint32_t l1AStages = 2;
constexpr uint32_t l1BStages = 2;
constexpr uint32_t l0AStages = 2;
constexpr uint32_t l0BStages = 2;
using DispatchPolicy = Gemm::MmadDequant<
    ArchTag, enableUnitFlag, useHF32, l0CStages, enableL1Resident, l1AStages, l1BStages, l0AStages, l0BStages>;
using L1TileShape = Shape<Int<256>, Int<256>, Int<128>>;
using L0TileShape = Shape<Int<256>, Int<256>, Int<32>>;

#if defined(CATLASS_JIT_QUANT_MODE) && (CATLASS_JIT_QUANT_MODE == 1)
constexpr Gemm::Tile::ScaleGranularity QuantMode = Gemm::Tile::ScaleGranularity::PER_CHANNEL;
#else
constexpr Gemm::Tile::ScaleGranularity QuantMode = Gemm::Tile::ScaleGranularity::PER_TENSOR;
#endif

// Preserve the original K; the device copy path handles the HiF8 tail.
using TileCopy = Gemm::Tile::PackedTileCopyTla<
    ArchTag, ElementA, LayoutTagA, ElementB, LayoutTagB, ElementC, LayoutTagC, ElementBias, false, QuantMode>;
using BlockMmad = Gemm::Block::BlockMmadTla<
    DispatchPolicy, L1TileShape, L0TileShape, ElementA, ElementB, ElementC, ElementBias, TileCopy>;
using BlockEpilogue = void;
using BlockScheduler = typename Gemm::Block::GemmIdentityBlockSwizzle<3, 1>;
using MatmulKernel = Gemm::Kernel::BatchedMatmulFixpipeDequantTla<BlockMmad, BlockScheduler>;

template <class LayoutTag>
static auto MakeBatchedLayout(uint32_t batch, uint32_t rows, uint32_t cols, int64_t batchStride)
{
    if constexpr (std::is_same_v<LayoutTag, layout::RowMajor>) {
        return tla::MakeLayout(
            tla::MakeShape(batch, rows, cols), tla::MakeStride(batchStride, static_cast<int64_t>(cols), tla::Int<1>{}));
    } else {
        return tla::MakeLayout(
            tla::MakeShape(batch, rows, cols), tla::MakeStride(batchStride, tla::Int<1>{}, static_cast<int64_t>(rows)));
    }
}

extern "C" void run(uint32_t blockNum, aclrtStream stream, const CatlassKernel::MatmulParams* params)
{
    uint32_t m = params->m;
    uint32_t n = params->n;
    uint32_t k = params->k;
    uint32_t batchCount = params->batch;

    GemmCoord problemShape{m, n, k};

    int64_t strideA = static_cast<int64_t>(m) * k;
    int64_t strideB = static_cast<int64_t>(k) * n;
    int64_t strideC = static_cast<int64_t>(m) * n;
    auto layoutA = MakeBatchedLayout<LayoutTagA>(batchCount, m, k, strideA);
    auto layoutB = MakeBatchedLayout<LayoutTagB>(batchCount, k, n, strideB);
    auto layoutC = MakeBatchedLayout<LayoutTagC>(batchCount, m, n, strideC);
    auto layoutBias = tla::MakeLayout(n);

    union {
        uint8_t* ptr;
        float val;
    } scaleUnion;
    scaleUnion.ptr = params->inputAddr[3];
    float scalePerTensor = scaleUnion.val;

    typename MatmulKernel::Arguments arguments{
        batchCount,
        problemShape,
        params->inputAddr[0],
        layoutA,
        params->inputAddr[1],
        layoutB,
        params->outputAddr[0],
        layoutC,
        scalePerTensor,
        params->inputAddr[2],
        params->inputAddr[4],
        layoutBias};

    Catlass::RunKernel<MatmulKernel>(arguments, stream, blockNum);
}
