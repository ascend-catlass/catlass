/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
 * BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE. See LICENSE in the root of
 * the software repository for the full text of the License.
 */

// By setting the K_MAX_SHAPE_DIM macro, the dimension of the AscendC Tensor's ShapeInfo is configured to 0,
// optimizing stack space. If you need to use the ShapeInfo of the AscendC Tensor, please undefine this macro.
#ifndef K_MAX_SHAPE_DIM
#define K_MAX_SHAPE_DIM 0
#endif

#include <iostream>
#include "catlass/gemm/kernel/svd_quant_matmul_tla.hpp"

#include <acl/acl.h>

#include "catlass/arch/arch.hpp"
#include "catlass/catlass.hpp"
#include "catlass/gemm/block/block_mmad.hpp"
#include "catlass/gemm/block/block_swizzle.hpp"
#include "catlass/gemm/device/device_gemm.hpp"
#include "catlass/gemm/dispatch_policy.hpp"
#include "catlass/gemm/gemm_type.hpp"
#include "catlass/layout/layout.hpp"
#include "catlass/status.hpp"
#include "tla/layout.hpp"

#include "golden.hpp"
#include "helper.hpp"

using namespace Catlass;
using namespace tla;

// run this example in the <catlass-project-root-path>
const std::string dataPath = "./examples/61_ascend950_svd_quant_matmul/data";

struct SvdQuantOptions {
    const std::string HELPER = "m n k r [device_id]";

    Catlass::GemmCoord problemShape{128, 128, 128};
    uint32_t problemRank{32};
    int32_t deviceId{0};

    SvdQuantOptions() = default;

    int Parse(int argc, const char** argv)
    {
        enum class ArgsIndex
        {
            M_INDEX = 1,
            N_INDEX,
            K_INDEX,
            R_INDEX,
            DEVICE_ID_INDEX,
            ARGS_MAX
        };

        if (argc > static_cast<uint32_t>(ArgsIndex::ARGS_MAX) ||
            argc < static_cast<uint32_t>(ArgsIndex::DEVICE_ID_INDEX)) {
            std::cerr << TOSTRING(CATLASS_EXAMPLE_NAME) << " " << HELPER << std::endl;
            return -1;
        }

        problemShape.m() = std::atoi(argv[static_cast<uint32_t>(ArgsIndex::M_INDEX)]);
        problemShape.n() = std::atoi(argv[static_cast<uint32_t>(ArgsIndex::N_INDEX)]);
        problemShape.k() = std::atoi(argv[static_cast<uint32_t>(ArgsIndex::K_INDEX)]);
        problemRank = std::atoi(argv[static_cast<uint32_t>(ArgsIndex::R_INDEX)]);
        if (argc == static_cast<uint32_t>(ArgsIndex::ARGS_MAX)) {
            deviceId = std::atoi(argv[static_cast<uint32_t>(ArgsIndex::DEVICE_ID_INDEX)]);
        }
        return 0;
    }
};
using Options = SvdQuantOptions;

enum class SvdQuantTilingTag
{
    Common,
    Small,
};
template <SvdQuantTilingTag TilingTag>
struct TilingTag2Config {};
template <>
struct TilingTag2Config<SvdQuantTilingTag::Common> {
    using L1TileShape1 = Shape<Int<128>, Int<128>, Int<256>>;
    using L0TileShape1 = Shape<Int<128>, Int<128>, Int<128>>;
    using L1TileShape2 = Shape<Int<256>, Int<256>, Int<128>>;
    using L0TileShape2 = Shape<Int<256>, Int<256>, Int<64>>;
    using L1TileShape3 = Shape<Int<256>, Int<256>, Int<512>>;
    using L0TileShape3 = Shape<Int<256>, Int<256>, Int<256>>;
};
template <>
struct TilingTag2Config<SvdQuantTilingTag::Small> {
    using L1TileShape1 = Shape<Int<128>, Int<128>, Int<256>>;
    using L0TileShape1 = Shape<Int<128>, Int<128>, Int<128>>;
    using L1TileShape2 = Shape<Int<128>, Int<256>, Int<128>>;
    using L0TileShape2 = Shape<Int<128>, Int<256>, Int<64>>;
    using L1TileShape3 = Shape<Int<128>, Int<256>, Int<512>>;
    using L0TileShape3 = Shape<Int<128>, Int<256>, Int<256>>;
};

template <SvdQuantTilingTag TilingTag>
static void Run(const Options& options)
{
    using InDType = half;
    using OutDType = half;
    using TilingConfig = TilingTag2Config<TilingTag>;

    aclrtStream stream{nullptr};

    ACL_CHECK(aclInit(nullptr));
    ACL_CHECK(aclrtSetDevice(options.deviceId));
    ACL_CHECK(aclrtCreateStream(&stream));

    uint32_t m = options.problemShape.m();
    uint32_t n = options.problemShape.n();
    uint32_t k = options.problemShape.k();
    uint32_t r = options.problemRank;
    uint32_t mxScaleK = CeilDiv<MX_SCALE_GROUP_NUM>(k);
    uint32_t mxScaleAlignedK = RoundUp<2>(mxScaleK);

    // qmax
    std::vector<float> hostQmax(1);
    ReadFile(dataPath + "/input/qmax.bin", hostQmax.data(), hostQmax.size() * sizeof(float));
    float qmax = hostQmax[0];

    // X
    using ElementX = InDType;
    using LayoutTagX = layout::RowMajor;
    size_t lenX = LayoutTagX::MakeLayout<ElementX>(m, k).Capacity();
    size_t sizeX = lenX * sizeof(ElementX);
    std::vector<fp16_t> hostX(lenX);
    ReadFile(dataPath + "/input/x.bin", hostX.data(), sizeX);
    auto layoutX = tla::MakeLayout<ElementX, LayoutTagX>(m, k);

    // Svd1
    using ElementSvd1 = InDType;
    using LayoutTagSvd1 = layout::ColumnMajor;
    size_t lenSvd1 = LayoutTagSvd1::MakeLayout<ElementSvd1>(k, r).Capacity();
    size_t sizeSvd1 = lenSvd1 * sizeof(ElementSvd1);
    std::vector<fp16_t> hostSvd1(lenSvd1);
    ReadFile(dataPath + "/input/svd1.bin", hostSvd1.data(), sizeSvd1);
    auto layoutSvd1 = tla::MakeLayout<ElementSvd1, LayoutTagSvd1>(k, r);

    // Svd2
    using ElementSvd2 = InDType;
    using LayoutTagSvd2 = layout::ColumnMajor;
    size_t lenSvd2 = LayoutTagSvd2::MakeLayout<ElementSvd2>(r, n).Capacity();
    size_t sizeSvd2 = lenSvd2 * sizeof(ElementSvd2);
    std::vector<fp16_t> hostSvd2(lenSvd2);
    ReadFile(dataPath + "/input/svd2.bin", hostSvd2.data(), sizeSvd2);
    auto layoutSvd2 = tla::MakeLayout<ElementSvd2, LayoutTagSvd2>(r, n);

    using ElementMxScale = float8_e8m0_t;
    // W
    using ElementW = float4_e2m1x2_t;
    using LayoutTagW = layout::ColumnMajor;
    size_t lenW = LayoutTagW::MakeLayout<ElementW>(k, n).Capacity();
    size_t sizeW = lenW / 2;
    std::vector<int8_t> hostW(lenW);
    ReadFile(dataPath + "/input/w.bin", hostW.data(), sizeW);
    auto layoutW = tla::MakeLayout<ElementW, LayoutTagW>(k, n);

    // MxScaleW
    size_t lenMxScaleW = mxScaleAlignedK * n;
    size_t sizeMxScaleW = lenMxScaleW * sizeof(ElementMxScale);
    std::vector<int8_t> hostMxScaleW(lenMxScaleW);
    ReadFile(dataPath + "/input/w_scale.bin", hostMxScaleW.data(), sizeMxScaleW);
    auto layoutMxScaleW = tla::MakeMxScaleLayout<ElementMxScale, LayoutTagW, true>(mxScaleK, n);

    // optional inputs
    // SmoothScale
    using ElementSmoothScale = InDType;
    using LayoutTagSmoothScale = layout::RowMajor;
    using GmSmoothType = std::conditional_t<std::is_void_v<ElementSmoothScale>, uint8_t, ElementSmoothScale>;
    size_t lenSmoothScale = k;
    size_t sizeSmoothScale = lenSmoothScale * sizeof(GmSmoothType);
    std::vector<fp16_t> hostSmoothScale(lenSmoothScale);
    if constexpr (!std::is_void_v<ElementSmoothScale>) {
        ReadFile(dataPath + "/input/smooth_scale.bin", hostSmoothScale.data(), sizeSmoothScale);
    }
    auto layoutSmoothScale = tla::MakeLayout<GmSmoothType, LayoutTagSmoothScale>((uint32_t)1, k);

    // optional input bias
    using ElementBias = void;
    size_t lenBias = n;
    using ElementBiasType = std::conditional_t<std::is_void_v<ElementBias>, uint8_t, ElementBias>;
    size_t sizeBias = n * sizeof(ElementBiasType);
    std::vector<ElementBiasType> hostBias(lenBias);
    if constexpr (!std::is_void_v<ElementBias>) {
        ReadFile(dataPath + "/input/bias.bin", hostBias.data(), sizeBias);
    }

    // workspace
    // C1
    using ElementC1 = InDType;
    using LayoutTagC1 = layout::RowMajor;
    // QuantX
    using ElementQuantX = float4_e2m1x2_t;
    using LayoutTagQuantX = LayoutTagX;
    // MxScaleX
    using ElementMxScaleX = float4_e2m1x2_t;
    auto layoutMxScaleX = tla::MakeMxScaleLayout<ElementMxScale, LayoutTagQuantX, false>(m, mxScaleK);

    // output Y
    using ElementY = OutDType;
    using LayoutTagY = layout::RowMajor;
    size_t lenY = LayoutTagY::MakeLayout<ElementY>(m, n).Capacity();
    size_t sizeY = lenY * sizeof(ElementY);
    auto layoutY = tla::MakeLayout<ElementY, LayoutTagY>(m, n);

    uint8_t* deviceX{nullptr};
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void**>(&deviceX), sizeX, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemcpy(deviceX, sizeX, hostX.data(), sizeX, ACL_MEMCPY_HOST_TO_DEVICE));

    uint8_t* deviceSvd1{nullptr};
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void**>(&deviceSvd1), sizeSvd1, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemcpy(deviceSvd1, sizeSvd1, hostSvd1.data(), sizeSvd1, ACL_MEMCPY_HOST_TO_DEVICE));

    uint8_t* deviceSvd2{nullptr};
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void**>(&deviceSvd2), sizeSvd2, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemcpy(deviceSvd2, sizeSvd2, hostSvd2.data(), sizeSvd2, ACL_MEMCPY_HOST_TO_DEVICE));

    uint8_t* deviceW{nullptr};
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void**>(&deviceW), sizeW, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemcpy(deviceW, sizeW, hostW.data(), sizeW, ACL_MEMCPY_HOST_TO_DEVICE));

    uint8_t* deviceMxScaleW{nullptr};
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void**>(&deviceMxScaleW), sizeMxScaleW, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemcpy(deviceMxScaleW, sizeMxScaleW, hostMxScaleW.data(), sizeMxScaleW, ACL_MEMCPY_HOST_TO_DEVICE));

    uint8_t* deviceSmoothScale{nullptr};
    if constexpr (!std::is_void_v<ElementSmoothScale>) {
        ACL_CHECK(
            aclrtMalloc(reinterpret_cast<void**>(&deviceSmoothScale), sizeSmoothScale, ACL_MEM_MALLOC_HUGE_FIRST));
        ACL_CHECK(aclrtMemcpy(
            deviceSmoothScale, sizeSmoothScale, hostSmoothScale.data(), sizeSmoothScale, ACL_MEMCPY_HOST_TO_DEVICE));
    }

    uint8_t* deviceBias{nullptr};
    if constexpr (!std::is_void_v<ElementBias>) {
        ACL_CHECK(aclrtMalloc(reinterpret_cast<void**>(&deviceBias), sizeBias, ACL_MEM_MALLOC_HUGE_FIRST));
        ACL_CHECK(aclrtMemcpy(deviceBias, sizeBias, hostBias.data(), sizeBias, ACL_MEMCPY_HOST_TO_DEVICE));
    }

    // output
    uint8_t* deviceY{nullptr};
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void**>(&deviceY), sizeY, ACL_MEM_MALLOC_HUGE_FIRST));

    using ArchTag = Arch::Ascend950;
    constexpr bool enableUnitFlag = true;

    // Mmad1
    using L1TileShape1 = typename TilingConfig::L1TileShape1;
    using L0TileShape1 = typename TilingConfig::L0TileShape1;
    using DispatchPolicy1 = Gemm::MmadSvd1<ArchTag, enableUnitFlag>;
    using TileCopy1 = Gemm::Tile::PackedTileCopyTla<
        ArchTag, ElementX, LayoutTagX, ElementSvd1, LayoutTagSvd1, ElementC1, LayoutTagC1, void>;
    using BlockMmad1 = Gemm::Block::BlockMmadTla<
        DispatchPolicy1, L1TileShape1, L0TileShape1, ElementX, ElementSvd1, ElementC1, void, TileCopy1>;

    // SmoothQuant
    using SmoothQuant = Gemm::Kernel::SmoothQuant<
        ArchTag, ElementX, ElementSmoothScale, ElementQuantX, ElementMxScale, LayoutTagX, L1TileShape1>;

    // Mmad2
    using L1TileShape2 = typename TilingConfig::L1TileShape2;
    using L0TileShape2 = typename TilingConfig::L0TileShape2;
    using DispatchPolicy2 = typename Gemm::MmadSvd2<ArchTag, enableUnitFlag>;
    using TileCopy2 = Gemm::Tile::PackedTileCopyTla<
        ArchTag, ElementC1, LayoutTagC1, ElementSvd2, LayoutTagSvd2, ElementY, LayoutTagY, ElementBias>;
    using BlockMmad2 = Gemm::Block::BlockMmadTla<
        DispatchPolicy2, L1TileShape2, L0TileShape2, ElementC1, ElementSvd2, ElementY, ElementBias, TileCopy2>;

    // Mmad3
    using L1TileShape3 = typename TilingConfig::L1TileShape3;
    using L0TileShape3 = typename TilingConfig::L0TileShape3;
    static constexpr uint32_t l1ScaleFactorK = 8;
    using DispatchPolicy3 = typename Gemm::MmadSvd3<ArchTag, enableUnitFlag, l1ScaleFactorK>;
    using TileCopy3 = Gemm::Tile::PackedMxTileCopyTla<
        ArchTag, ElementQuantX, LayoutTagQuantX, ElementW, LayoutTagW, ElementMxScale, decltype(layoutMxScaleX),
        ElementMxScale, decltype(layoutMxScaleW), ElementY, LayoutTagY, void>;
    using BlockMmad3 = Gemm::Block::BlockMmadTla<
        DispatchPolicy3, L1TileShape3, L0TileShape3, ElementQuantX, ElementW, ElementY, void, TileCopy3>;

    using BlockEpilogue = void;

    // Get the number of cube cores of the current hardware
    auto aicCoreNum = platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreNumAic();
    uint32_t taskNum1 = CeilDiv(m, tla::get<0>(L1TileShape1{})) * CeilDiv(r, tla::get<1>(L1TileShape1{}));
    uint32_t taskNum2 = CeilDiv(m, tla::get<0>(L1TileShape3{})) * CeilDiv(n, tla::get<1>(L1TileShape3{}));
    uint32_t aicCoreUsed = min(aicCoreNum, max(taskNum1, taskNum2));

    // Swizzle offset is 3 and direction is 0.
    using BlockScheduler = typename Gemm::Block::GemmIdentityBlockSwizzle<3, 0>;

    // kernel level
    using MatmulKernel =
        Gemm::Kernel::SvdQuantMatmulTla<SmoothQuant, BlockMmad1, BlockMmad2, BlockMmad3, BlockEpilogue, BlockScheduler>;
    typename MatmulKernel::Arguments arguments{
        options.problemShape, options.problemRank, qmax,
        // inputs
        deviceX, layoutX, deviceSvd1, layoutSvd1, deviceSvd2, layoutSvd2, deviceW, layoutW, deviceMxScaleW,
        layoutMxScaleW,
        // optional inputs
        deviceSmoothScale, layoutSmoothScale, deviceBias,
        // output
        deviceY, layoutY};
    using MatmulAdapter = Gemm::Device::DeviceGemm<MatmulKernel>;
    MatmulAdapter matmulOp;

    matmulOp.CanImplement(arguments);
    uint8_t* deviceWorkspace{nullptr};
    size_t sizeWorkspace = matmulOp.GetWorkspaceSize(arguments);
    if (sizeWorkspace > 0) {
        ACL_CHECK(aclrtMalloc(reinterpret_cast<void**>(&deviceWorkspace), sizeWorkspace, ACL_MEM_MALLOC_HUGE_FIRST));
    }
    matmulOp.Initialize(arguments, deviceWorkspace);
    matmulOp(stream, aicCoreUsed);

    ACL_CHECK(aclrtSynchronizeStream(stream));

    // Compare output Y
    std::vector<fp16_t> hostY(lenY);
    ACL_CHECK(aclrtMemcpy(hostY.data(), sizeY, deviceY, sizeY, ACL_MEMCPY_DEVICE_TO_HOST));

    std::vector<float> hostGolden(lenY);
    ReadFile(dataPath + "/golden/y_golden.bin", hostGolden.data(), sizeof(float) * lenY);
    std::vector<float> hostCpu(lenY);
    ReadFile(dataPath + "/golden/y_cpu.bin", hostCpu.data(), sizeof(float) * lenY);

    auto errorMetrics = golden::ComputeErrorMetrics(hostY, hostCpu, hostGolden);

    // Element-wise comparison: rtol depends on K (accumulation dimension)
    std::vector<uint64_t> errorIndices = golden::CompareData(hostY, hostCpu, k);
    if (errorIndices.empty() || errorMetrics.passed) {
        std::cout << "Compare success." << std::endl;
    } else {
        std::cerr << "Compare failed. Error count: " << errorIndices.size() << std::endl;
        std::cerr << "Error ratios exceed thresholds:" << std::endl;
        std::cerr << "MARE ratio: " << errorMetrics.mareRatio << " (threshold: 5)" << std::endl;
        std::cerr << "MERE ratio: " << errorMetrics.mereRatio << " (threshold: 1.5)" << std::endl;
        std::cerr << "RMSE ratio: " << errorMetrics.rmseRatio << " (threshold: 1.5)" << std::endl;
    }

    ACL_CHECK(aclrtFree(deviceX));
    ACL_CHECK(aclrtFree(deviceSvd1));
    ACL_CHECK(aclrtFree(deviceSvd2));
    ACL_CHECK(aclrtFree(deviceW));
    ACL_CHECK(aclrtFree(deviceMxScaleW));
    ACL_CHECK(aclrtFree(deviceY));
    if constexpr (!std::is_void_v<ElementSmoothScale>) {
        ACL_CHECK(aclrtFree(deviceSmoothScale));
    }
    if constexpr (!std::is_void_v<ElementBias>) {
        ACL_CHECK(aclrtFree(deviceBias));
    }
    if (sizeWorkspace > 0) {
        ACL_CHECK(aclrtFree(deviceWorkspace));
    }

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
    uint32_t m = options.problemShape.m();
    uint32_t n = options.problemShape.n();
    uint32_t blocks = CeilDiv(m, 256) * CeilDiv(n, 256);
    auto aicCoreNum = platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreNumAic();

    if (blocks < aicCoreNum) {
        Run<SvdQuantTilingTag::Small>(options);
    } else {
        Run<SvdQuantTilingTag::Common>(options);
    }
    return 0;
}
