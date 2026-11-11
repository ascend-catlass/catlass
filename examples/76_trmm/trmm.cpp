/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// By setting the K_MAX_SHAPE_DIM macro, the dimension of the AscendC Tensor's ShapeInfo is configured to 0,
// optimizing stack space. If you need to use the ShapeInfo of the AscendC Tensor, please undefine this macro.
#ifndef K_MAX_SHAPE_DIM
#define K_MAX_SHAPE_DIM 0
#endif

#include <iostream>
#include "catlass/gemm/kernel/trmm.hpp"

#include "catlass/arch/arch.hpp"
#include "catlass/catlass.hpp"
#include "catlass/gemm/block/block_mmad.hpp"
#include "catlass/gemm/block/block_swizzle.hpp"
#include "catlass/gemm/device/device_gemm.hpp"
#include "catlass/gemm/dispatch_policy.hpp"
#include "catlass/layout/layout.hpp"
#include "catlass/status.hpp"
#include "tla/layout.hpp"
#include "tla/tensor.hpp"

#include "golden.hpp"
#include "helper.hpp"

#include <vector>

using namespace Catlass;

using Options = TrmmOptions;

// Tile/Swizzle configurations matching TRMM delivery package
using TrmmDefaultL1 = tla::tuple<tla::C<128>, tla::C<128>, tla::C<256>>;
using TrmmDefaultL0 = tla::tuple<tla::C<128>, tla::C<128>, tla::C<64>>;
using TrmmDefaultSwizzle30 = Gemm::Block::GemmIdentityBlockSwizzle<3, 0>;
using TrmmDefaultSwizzle31 = Gemm::Block::GemmIdentityBlockSwizzle<3, 1>;

using TrmmSRLowerL1 = tla::tuple<tla::C<128>, tla::C<128>, tla::C<64>>;
using TrmmSRLowerL0 = tla::tuple<tla::C<128>, tla::C<128>, tla::C<64>>;
using TrmmSRLowerSwizzle = Gemm::Block::GemmIdentityBlockSwizzle<1, 0>;

using TrmmSRUpperL1 = tla::tuple<tla::C<128>, tla::C<128>, tla::C<64>>;
using TrmmSRUpperL0 = tla::tuple<tla::C<128>, tla::C<128>, tla::C<64>>;
using TrmmSRUpperSwizzle = Gemm::Block::GemmIdentityBlockSwizzle<4, 1>;

using TrmmSRK128L1 = tla::tuple<tla::C<128>, tla::C<128>, tla::C<128>>;
using TrmmSRK128L0 = tla::tuple<tla::C<128>, tla::C<128>, tla::C<64>>;
using TrmmSRK128Swizzle = Gemm::Block::GemmIdentityBlockSwizzle<1, 1>;

using TrmmSLK128L1 = tla::tuple<tla::C<128>, tla::C<128>, tla::C<128>>;
using TrmmSLK128L0 = tla::tuple<tla::C<128>, tla::C<128>, tla::C<64>>;
using TrmmSLK128Swizzle = Gemm::Block::GemmIdentityBlockSwizzle<1, 1>;

constexpr uint32_t SMALL_RIGHT_N_LIMIT = 512u;
constexpr uint32_t SMALL_RIGHT_K128_N_LIMIT = 256u;
constexpr uint32_t SMALL_LEFT_M_LIMIT = 512u;
constexpr uint32_t SMALL_LEFT_K128_M_LIMIT = 256u;

enum class TileVariant : uint64_t
{
    TILE_DEFAULT = 0,
    TILE_SMALL_RIGHT_LOWER = 1,
    TILE_SMALL_RIGHT_UPPER = 2,
    TILE_SMALL_RIGHT_K128 = 3,
    TILE_DEFAULT_SWIZZLE31 = 4,
    TILE_SMALL_LEFT_LOWER = 5,
    TILE_SMALL_LEFT_UPPER = 6,
    TILE_SMALL_LEFT_K128 = 7,
};

static TileVariant SelectTileVariant(uint32_t side, uint32_t uplo, uint32_t trans, uint32_t m, uint32_t n)
{
    uint32_t effectiveUplo = uplo ^ trans;
    bool isUpper = (effectiveUplo == 1u);
    bool isLower = (effectiveUplo == 0u);

    if (side == 1) {
        if (isUpper && n <= SMALL_RIGHT_K128_N_LIMIT)
            return TileVariant::TILE_SMALL_RIGHT_K128;
        if (isUpper && n <= SMALL_RIGHT_N_LIMIT)
            return TileVariant::TILE_SMALL_RIGHT_UPPER;
        if (isLower && n <= SMALL_RIGHT_N_LIMIT)
            return TileVariant::TILE_SMALL_RIGHT_LOWER;
    } else {
        if (isUpper && m <= SMALL_LEFT_K128_M_LIMIT)
            return TileVariant::TILE_SMALL_LEFT_K128;
        if (isUpper && m <= SMALL_LEFT_M_LIMIT)
            return TileVariant::TILE_SMALL_LEFT_UPPER;
        if (isLower && m <= SMALL_LEFT_M_LIMIT)
            return TileVariant::TILE_SMALL_LEFT_LOWER;
    }

    if (m >= 2048 && m < n)
        return TileVariant::TILE_DEFAULT_SWIZZLE31;
    return TileVariant::TILE_DEFAULT;
}

template <class LayoutTagA, class LayoutTagB, class L1TileShape, class L0TileShape, class BlockScheduler>
static bool RunTrmmDispatch(
    GemmCoord problemShape, uint8_t* deviceA, uint8_t* deviceB, uint8_t* deviceC, uint32_t side, uint32_t uplo,
    uint32_t trans, float alpha, aclrtStream stream, uint32_t aicCoreNum)
{
    using ElementA = float;
    using ElementB = float;
    using ElementC = float;
    using LayoutTagC = layout::RowMajor;
    using ArchTag = Arch::AtlasA2;
    constexpr bool enableUnitFlag = true;
    constexpr bool useHF32 = false;
    using DispatchPolicy = Gemm::MmadPingpong<ArchTag, enableUnitFlag, useHF32>;

    using TileCopy =
        Gemm::Tile::PackedTileCopyTla<ArchTag, ElementA, LayoutTagA, ElementB, LayoutTagB, ElementC, LayoutTagC>;
    using BlockMmad = Gemm::Block::BlockMmadTla<
        DispatchPolicy, L1TileShape, L0TileShape, ElementA, ElementB, ElementC, void, TileCopy>;
    using BlockEpilogue = void;

    using TrmmKernel = Gemm::Kernel::Trmm<BlockMmad, BlockEpilogue, BlockScheduler>;
    using TrmmAdapter = Gemm::Device::DeviceGemm<TrmmKernel>;

    auto layoutA = tla::MakeLayout<ElementA, LayoutTagA>(problemShape.m(), problemShape.k());
    auto layoutB = tla::MakeLayout<ElementB, LayoutTagB>(problemShape.k(), problemShape.n());
    auto layoutC = tla::MakeLayout<ElementC, LayoutTagC>(problemShape.m(), problemShape.n());

    typename TrmmKernel::Arguments arguments{problemShape, deviceA, layoutA, deviceB, layoutB, deviceC,
                                             layoutC,      side,    uplo,    trans,   0u,      alpha};
    TrmmAdapter trmmOp;
    if (trmmOp.CanImplement(arguments) != Status::kSuccess) {
        std::cerr << "[ERROR] TRMM kernel cannot implement the supplied arguments." << std::endl;
        return false;
    }
    size_t sizeWorkspace = trmmOp.GetWorkspaceSize(arguments);
    uint8_t* deviceWorkspace = nullptr;
    if (sizeWorkspace > 0) {
        ACL_CHECK(aclrtMalloc(reinterpret_cast<void**>(&deviceWorkspace), sizeWorkspace, ACL_MEM_MALLOC_HUGE_FIRST));
    }
    if (trmmOp.Initialize(arguments, deviceWorkspace) != Status::kSuccess) {
        std::cerr << "[ERROR] Failed to initialize TRMM kernel." << std::endl;
        if (deviceWorkspace != nullptr) {
            ACL_CHECK(aclrtFree(deviceWorkspace));
        }
        return false;
    }
    trmmOp(stream, aicCoreNum);
    ACL_CHECK(aclrtSynchronizeStream(stream));

    if (sizeWorkspace > 0) {
        ACL_CHECK(aclrtFree(deviceWorkspace));
    }
    return true;
}

#define TRMM_DISPATCH(LayoutA_t, LayoutB_t, L1_t, L0_t, Swizzle_t)                    \
    dispatchSucceeded = RunTrmmDispatch<LayoutA_t, LayoutB_t, L1_t, L0_t, Swizzle_t>( \
        problemShape, deviceA, deviceB, deviceC, side, uplo, trans, kernelAlpha, stream, aicCoreNum)

static void Run(const Options& options)
{
    aclrtStream stream{nullptr};

    ACL_CHECK(aclInit(nullptr));
    ACL_CHECK(aclrtSetDevice(options.deviceId));
    ACL_CHECK(aclrtCreateStream(&stream));

    uint32_t m = options.problemShape.m();
    uint32_t n = options.problemShape.n();
    uint32_t k = options.problemShape.k();
    uint32_t side = options.side;
    uint32_t uplo = options.uplo;
    uint32_t trans = options.trans;
    uint32_t diag = options.diag;
    float alpha = options.alpha;

    if (m == 0 || n == 0) {
        std::cerr << "[ERROR] Dimensions M and N must be greater than 0." << std::endl;
        return;
    }
    if (side > 1) {
        std::cerr << "[ERROR] Invalid side: " << side << ". Must be 0 (left) or 1 (right)." << std::endl;
        return;
    }
    if (uplo > 1) {
        std::cerr << "[ERROR] Invalid uplo: " << uplo << ". Must be 0 (lower) or 1 (upper)." << std::endl;
        return;
    }
    if (trans > 1) {
        std::cerr << "[ERROR] Invalid trans: " << trans << ". Must be 0 (non-transpose) or 1 (transpose)." << std::endl;
        return;
    }
    if (diag != 0) {
        std::cerr << "[ERROR] Invalid diag: " << diag << ". This example currently supports only 0 (non-unit)."
                  << std::endl;
        return;
    }

    size_t lenA = static_cast<size_t>(m) * k;
    size_t lenB = static_cast<size_t>(k) * n;
    size_t lenC = static_cast<size_t>(m) * n;

    size_t sizeA = lenA * sizeof(float);
    size_t sizeB = lenB * sizeof(float);
    size_t sizeC = lenC * sizeof(float);

    // Layout dispatch matching TRMM:
    //   trans=0: Row/Row
    //   trans=1, side=left: Col/Row
    //   trans=1, side=right: Row/Col
    using LayoutRowRowA = layout::RowMajor;
    using LayoutRowRowB = layout::RowMajor;
    using LayoutColRowA = layout::ColumnMajor;
    using LayoutColRowB = layout::RowMajor;
    using LayoutRowColA = layout::RowMajor;
    using LayoutRowColB = layout::ColumnMajor;

    uint64_t triElements = static_cast<uint64_t>(k) * static_cast<uint64_t>(k);
    uint64_t outputElements = static_cast<uint64_t>(m) * static_cast<uint64_t>(n);
    bool fuseAlphaInPrepare = (triElements <= outputElements);
    float prepareAlpha = fuseAlphaInPrepare ? alpha : 1.0f;
    float kernelAlpha = fuseAlphaInPrepare ? 1.0f : alpha;

    std::vector<float> hostA(lenA);
    std::vector<float> hostB(lenB);
    if (side == 0) {
        golden::FillTriangularData(hostA, m, k, uplo, diag, 0.5f, 1.5f, prepareAlpha);
        golden::FillRandomData(hostB, 0.5f, 1.5f);
    } else {
        golden::FillRandomData(hostA, 0.5f, 1.5f);
        golden::FillTriangularData(hostB, k, n, uplo, diag, 0.5f, 1.5f, prepareAlpha);
    }

    uint8_t* deviceA{nullptr};
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void**>(&deviceA), sizeA, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemcpy(deviceA, sizeA, hostA.data(), sizeA, ACL_MEMCPY_HOST_TO_DEVICE));

    uint8_t* deviceB{nullptr};
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void**>(&deviceB), sizeB, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemcpy(deviceB, sizeB, hostB.data(), sizeB, ACL_MEMCPY_HOST_TO_DEVICE));

    uint8_t* deviceC{nullptr};
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void**>(&deviceC), sizeC, ACL_MEM_MALLOC_HUGE_FIRST));

    auto aicCoreNum = platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreNumAic();

    GemmCoord problemShape{m, n, k};
    TileVariant tileVariant = SelectTileVariant(side, uplo, trans, m, n);
    bool dispatchSucceeded = false;

    // Dispatch matching TRMM TilingKey logic
    if (trans == 0) {
        // === layout = Row/Row (trans=0) ===
        switch (tileVariant) {
            case TileVariant::TILE_DEFAULT:
                TRMM_DISPATCH(LayoutRowRowA, LayoutRowRowB, TrmmDefaultL1, TrmmDefaultL0, TrmmDefaultSwizzle30);
                break;
            case TileVariant::TILE_DEFAULT_SWIZZLE31:
                TRMM_DISPATCH(LayoutRowRowA, LayoutRowRowB, TrmmDefaultL1, TrmmDefaultL0, TrmmDefaultSwizzle31);
                break;
            case TileVariant::TILE_SMALL_RIGHT_LOWER:
                TRMM_DISPATCH(LayoutRowRowA, LayoutRowRowB, TrmmSRLowerL1, TrmmSRLowerL0, TrmmSRLowerSwizzle);
                break;
            case TileVariant::TILE_SMALL_RIGHT_UPPER:
                TRMM_DISPATCH(LayoutRowRowA, LayoutRowRowB, TrmmSRUpperL1, TrmmSRUpperL0, TrmmSRUpperSwizzle);
                break;
            case TileVariant::TILE_SMALL_RIGHT_K128:
                TRMM_DISPATCH(LayoutRowRowA, LayoutRowRowB, TrmmSRK128L1, TrmmSRK128L0, TrmmSRK128Swizzle);
                break;
            case TileVariant::TILE_SMALL_LEFT_LOWER:
                TRMM_DISPATCH(LayoutRowRowA, LayoutRowRowB, TrmmDefaultL1, TrmmDefaultL0, TrmmDefaultSwizzle30);
                break;
            case TileVariant::TILE_SMALL_LEFT_UPPER:
                TRMM_DISPATCH(LayoutRowRowA, LayoutRowRowB, TrmmDefaultL1, TrmmDefaultL0, TrmmDefaultSwizzle30);
                break;
            case TileVariant::TILE_SMALL_LEFT_K128:
                TRMM_DISPATCH(LayoutRowRowA, LayoutRowRowB, TrmmSLK128L1, TrmmSLK128L0, TrmmSLK128Swizzle);
                break;
        }
    } else if (trans == 1 && side == 0) {
        // === layout = Col/Row (trans=1, side=left) ===
        switch (tileVariant) {
            case TileVariant::TILE_DEFAULT:
                TRMM_DISPATCH(LayoutColRowA, LayoutColRowB, TrmmDefaultL1, TrmmDefaultL0, TrmmDefaultSwizzle30);
                break;
            case TileVariant::TILE_DEFAULT_SWIZZLE31:
                TRMM_DISPATCH(LayoutColRowA, LayoutColRowB, TrmmDefaultL1, TrmmDefaultL0, TrmmDefaultSwizzle31);
                break;
            case TileVariant::TILE_SMALL_LEFT_LOWER:
                TRMM_DISPATCH(LayoutColRowA, LayoutColRowB, TrmmDefaultL1, TrmmDefaultL0, TrmmDefaultSwizzle30);
                break;
            case TileVariant::TILE_SMALL_LEFT_UPPER:
                TRMM_DISPATCH(LayoutColRowA, LayoutColRowB, TrmmDefaultL1, TrmmDefaultL0, TrmmDefaultSwizzle30);
                break;
            case TileVariant::TILE_SMALL_LEFT_K128:
                TRMM_DISPATCH(LayoutColRowA, LayoutColRowB, TrmmSLK128L1, TrmmSLK128L0, TrmmSLK128Swizzle);
                break;
            default:
                TRMM_DISPATCH(LayoutColRowA, LayoutColRowB, TrmmDefaultL1, TrmmDefaultL0, TrmmDefaultSwizzle30);
                break;
        }
    } else {
        // === layout = Row/Col (trans=1, side=right) ===
        switch (tileVariant) {
            case TileVariant::TILE_DEFAULT:
                TRMM_DISPATCH(LayoutRowColA, LayoutRowColB, TrmmDefaultL1, TrmmDefaultL0, TrmmDefaultSwizzle30);
                break;
            case TileVariant::TILE_DEFAULT_SWIZZLE31:
                TRMM_DISPATCH(LayoutRowColA, LayoutRowColB, TrmmDefaultL1, TrmmDefaultL0, TrmmDefaultSwizzle31);
                break;
            case TileVariant::TILE_SMALL_RIGHT_LOWER:
                TRMM_DISPATCH(LayoutRowColA, LayoutRowColB, TrmmSRLowerL1, TrmmSRLowerL0, TrmmSRLowerSwizzle);
                break;
            case TileVariant::TILE_SMALL_RIGHT_UPPER:
                TRMM_DISPATCH(LayoutRowColA, LayoutRowColB, TrmmSRUpperL1, TrmmSRUpperL0, TrmmSRUpperSwizzle);
                break;
            case TileVariant::TILE_SMALL_RIGHT_K128:
                TRMM_DISPATCH(LayoutRowColA, LayoutRowColB, TrmmSRK128L1, TrmmSRK128L0, TrmmSRK128Swizzle);
                break;
            default:
                TRMM_DISPATCH(LayoutRowColA, LayoutRowColB, TrmmDefaultL1, TrmmDefaultL0, TrmmDefaultSwizzle30);
                break;
        }
    }

    if (!dispatchSucceeded) {
        ACL_CHECK(aclrtFree(deviceA));
        ACL_CHECK(aclrtFree(deviceB));
        ACL_CHECK(aclrtFree(deviceC));
        ACL_CHECK(aclrtDestroyStream(stream));
        ACL_CHECK(aclrtResetDevice(options.deviceId));
        ACL_CHECK(aclFinalize());
        return;
    }

    std::vector<float> hostC(lenC);
    ACL_CHECK(aclrtMemcpy(hostC.data(), sizeC, deviceC, sizeC, ACL_MEMCPY_DEVICE_TO_HOST));

    std::vector<float> hostGolden(lenC);
    if (trans == 0) {
        golden::ComputeMatmul(
            problemShape, hostA, layout::RowMajor(m, k), hostB, layout::RowMajor(k, n), hostGolden,
            layout::RowMajor(m, n));
    } else if (side == 0) {
        golden::ComputeMatmul(
            problemShape, hostA, layout::ColumnMajor(m, k), hostB, layout::RowMajor(k, n), hostGolden,
            layout::RowMajor(m, n));
    } else {
        golden::ComputeMatmul(
            problemShape, hostA, layout::RowMajor(m, k), hostB, layout::ColumnMajor(k, n), hostGolden,
            layout::RowMajor(m, n));
    }
    if (kernelAlpha != 1.0f) {
        for (auto& v : hostGolden) {
            v *= kernelAlpha;
        }
    }
    std::vector<uint64_t> errorIndices = golden::CompareData(hostC, hostGolden, k);
    if (errorIndices.empty()) {
        std::cout << "Compare success." << std::endl;
    } else {
        std::cerr << "Compare failed. Error count: " << errorIndices.size() << std::endl;
    }

    ACL_CHECK(aclrtFree(deviceA));
    ACL_CHECK(aclrtFree(deviceB));
    ACL_CHECK(aclrtFree(deviceC));

    ACL_CHECK(aclrtDestroyStream(stream));
    ACL_CHECK(aclrtResetDevice(options.deviceId));
    ACL_CHECK(aclFinalize());
}

int main(int argc, const char** argv)
{
    Options options;
    if (options.Parse(argc, argv) != 0) {
        return -1;
    }
    Run(options);
    return 0;
}
