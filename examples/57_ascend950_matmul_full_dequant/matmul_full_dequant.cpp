/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
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
#include "catlass/gemm/kernel/matmul_full_dequant.hpp"

#include "catlass/arch/arch.hpp"
#include "catlass/catlass.hpp"
#include "catlass/epilogue/block/block_epilogue.hpp"
#include "catlass/epilogue/block/block_epilogue_dequant.hpp"
#include "catlass/epilogue/dispatch_policy.hpp"
#include "catlass/gemm/block/block_mmad.hpp"
#include "catlass/gemm/block/block_scheduler_aswt.hpp"
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

using Options = GemmOptions;
using QuantMode = Epilogue::Block::QuantMode;

struct MatmulShape {
    uint32_t m;
    uint32_t n;
    uint32_t k;
};

static void Run(const Options& options, QuantMode x1QuantMode, QuantMode x2QuantMode, bool hasQuantBias)
{
    aclrtStream stream{nullptr};

    ACL_CHECK(aclInit(nullptr));
    ACL_CHECK(aclrtSetDevice(options.deviceId));
    ACL_CHECK(aclrtCreateStream(&stream));

    uint32_t m = options.problemShape.m();
    uint32_t n = options.problemShape.n();
    uint32_t k = options.problemShape.k();

    using ElementA = int8_t;
    using ElementB = int8_t;
    using ElementC = int32_t;
    using ElementD = half;
    using ElementX1 = float;
    using ElementX2 = float;
    using ElementBias = float;

    using LayoutTagA = layout::RowMajor;
    using LayoutTagB = layout::RowMajor;
    using LayoutTagC = layout::RowMajor;
    LayoutTagA tagA = LayoutTagA::MakeLayout<ElementA>(m, k);
    LayoutTagB tagB = LayoutTagB::MakeLayout<ElementB>(k, n);
    LayoutTagC tagC = LayoutTagC::MakeLayout<ElementC>(m, n);

    size_t lenA = tagA.Capacity();
    size_t lenB = tagB.Capacity();
    size_t lenC = tagC.Capacity();
    size_t lenBias = static_cast<size_t>(n);

    size_t sizeA = lenA * sizeof(ElementA);
    size_t sizeB = lenB * sizeof(ElementB);
    size_t sizeC = lenC * sizeof(fp16_t);
    size_t sizeBias = lenBias * sizeof(ElementBias);
    size_t goldenSize = lenC * sizeof(float);
    size_t sizeWorkspace = 0;

    uint8_t* deviceA{nullptr};
    uint8_t* deviceB{nullptr};
    uint8_t* deviceC{nullptr};
    uint8_t* deviceWorkspace{nullptr};
    uint8_t* x1ScaleDevice{nullptr};
    uint8_t* x2ScaleDevice{nullptr};
    uint8_t* quantBiasDevice{nullptr};
    uint8_t* hostA{nullptr};
    uint8_t* hostB{nullptr};
    uint8_t* x1ScaleHost{nullptr};
    uint8_t* x2ScaleHost{nullptr};
    uint8_t* quantBiasHost{nullptr};

    const auto cleanupAndFinalize = [&]() {
        if (deviceA)
            ACL_CHECK(aclrtFree(deviceA));
        if (deviceB)
            ACL_CHECK(aclrtFree(deviceB));
        if (x1ScaleDevice)
            ACL_CHECK(aclrtFree(x1ScaleDevice));
        if (x2ScaleDevice)
            ACL_CHECK(aclrtFree(x2ScaleDevice));
        if (quantBiasDevice)
            ACL_CHECK(aclrtFree(quantBiasDevice));
        if (deviceC)
            ACL_CHECK(aclrtFree(deviceC));
        if (sizeWorkspace > 0 && deviceWorkspace)
            ACL_CHECK(aclrtFree(deviceWorkspace));
        if (hostA)
            ACL_CHECK(aclrtFreeHost(hostA));
        if (hostB)
            ACL_CHECK(aclrtFreeHost(hostB));
        if (x1ScaleHost)
            ACL_CHECK(aclrtFreeHost(x1ScaleHost));
        if (x2ScaleHost)
            ACL_CHECK(aclrtFreeHost(x2ScaleHost));
        if (quantBiasHost)
            ACL_CHECK(aclrtFreeHost(quantBiasHost));
        ACL_CHECK(aclrtDestroyStream(stream));
        ACL_CHECK(aclrtResetDevice(options.deviceId));
        ACL_CHECK(aclFinalize());
    };

    ACL_CHECK(aclrtMallocHost((void**)(&hostA), sizeA));
    if (!ReadFile("./input/x1.bin", hostA, sizeA)) {
        cleanupAndFinalize();
        return;
    }
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void**>(&deviceA), sizeA, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemcpy(deviceA, sizeA, hostA, sizeA, ACL_MEMCPY_HOST_TO_DEVICE));

    ACL_CHECK(aclrtMallocHost((void**)(&hostB), sizeB));
    if (!ReadFile("./input/x2.bin", hostB, sizeB)) {
        cleanupAndFinalize();
        return;
    }
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void**>(&deviceB), sizeB, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemcpy(deviceB, sizeB, hostB, sizeB, ACL_MEMCPY_HOST_TO_DEVICE));

    ACL_CHECK(aclrtMalloc(reinterpret_cast<void**>(&deviceC), sizeC, ACL_MEM_MALLOC_HUGE_FIRST));

    size_t x1ScaleFileSize = 0;
    if (x1QuantMode == QuantMode::PERTENSOR_MODE) {
        x1ScaleFileSize = sizeof(float);
    } else if (x1QuantMode == QuantMode::PERTOKEN_MODE) {
        x1ScaleFileSize = m * sizeof(float);
    }
    if (x1QuantMode != QuantMode::DEFAULT) {
        ACL_CHECK(aclrtMallocHost((void**)(&x1ScaleHost), x1ScaleFileSize));
        ACL_CHECK(aclrtMalloc((void**)&x1ScaleDevice, x1ScaleFileSize, ACL_MEM_MALLOC_HUGE_FIRST));
        if (!ReadFile("./input/x1_scale.bin", x1ScaleHost, x1ScaleFileSize)) {
            cleanupAndFinalize();
            return;
        }
        ACL_CHECK(aclrtMemcpy(x1ScaleDevice, x1ScaleFileSize, x1ScaleHost, x1ScaleFileSize, ACL_MEMCPY_HOST_TO_DEVICE));
    }

    size_t x2ScaleFileSize = 0;
    if (x2QuantMode == QuantMode::PERTENSOR_MODE) {
        x2ScaleFileSize = sizeof(float);
    } else if (x2QuantMode == QuantMode::PERCHANNEL_MODE) {
        x2ScaleFileSize = n * sizeof(float);
    }
    if (x2QuantMode != QuantMode::DEFAULT) {
        ACL_CHECK(aclrtMallocHost((void**)(&x2ScaleHost), x2ScaleFileSize));
        ACL_CHECK(aclrtMalloc((void**)&x2ScaleDevice, x2ScaleFileSize, ACL_MEM_MALLOC_HUGE_FIRST));
        if (!ReadFile("./input/x2_scale.bin", x2ScaleHost, x2ScaleFileSize)) {
            cleanupAndFinalize();
            return;
        }
        ACL_CHECK(aclrtMemcpy(x2ScaleDevice, x2ScaleFileSize, x2ScaleHost, x2ScaleFileSize, ACL_MEMCPY_HOST_TO_DEVICE));
    }

    if (hasQuantBias) {
        size_t quantBiasFileSize = n * sizeof(float);
        ACL_CHECK(aclrtMallocHost((void**)(&quantBiasHost), quantBiasFileSize));
        ACL_CHECK(aclrtMalloc((void**)&quantBiasDevice, quantBiasFileSize, ACL_MEM_MALLOC_HUGE_FIRST));
        if (!ReadFile("./input/bias.bin", quantBiasHost, quantBiasFileSize)) {
            cleanupAndFinalize();
            return;
        }
        ACL_CHECK(aclrtMemcpy(
            quantBiasDevice, quantBiasFileSize, quantBiasHost, quantBiasFileSize, ACL_MEMCPY_HOST_TO_DEVICE));
    }

    // Get the number of cube cores of the current hardware
    auto aicCoreNum = platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreNumAic();

    using ArchTag = Arch::Ascend950;
    constexpr bool enableUnitFlag = true;
    using DispatchPolicy = Gemm::MmadPingpong<ArchTag, enableUnitFlag>;

    using L1TileShape = Shape<Int<128>, Int<256>, Int<256>>;
    using L0TileShape = Shape<Int<128>, Int<256>, Int<128>>;

    auto layoutA = tla::MakeLayout<ElementA, LayoutTagA>(m, k);
    auto layoutB = tla::MakeLayout<ElementB, LayoutTagB>(k, n);
    auto layoutC = tla::MakeLayout<ElementC, LayoutTagC>(m, n);

    using ProblemShape = MatmulShape;

    MatmulShape shape = {m, n, k};

    using TileCopy = Gemm::Tile::PackedTileCopyTlaToUB<
        ArchTag, ElementA, LayoutTagA, ElementB, LayoutTagB, ElementC, LayoutTagC, ElementBias,
        Gemm::Tile::CopyL0CToUBMode::SPLIT_M>;
    using BlockMmad = Gemm::Block::BlockMmadTla<
        DispatchPolicy, L1TileShape, L0TileShape, ElementA, ElementB, ElementC, ElementBias, TileCopy>;
    using EpilogueDispatchPolicy = Epilogue::BlockEpilogueDequant;
    using BlockEpilogue = Epilogue::Block::BlockEpilogue<
        EpilogueDispatchPolicy, L0TileShape, ElementD, ElementC, ElementX1, ElementX2, ElementBias>;

    uint32_t taskNum = CeilDiv(options.problemShape.m(), tla::get<0>(L1TileShape{})) *
                       CeilDiv(options.problemShape.n(), tla::get<1>(L1TileShape{}));
    uint32_t aicCoreUsed = min(aicCoreNum, taskNum);

    using BlockScheduler = typename Gemm::Block::BlockSchedulerAswt<L1TileShape, L0TileShape>;

    // kernel level
    using MatmulKernel = Gemm::Kernel::GemmUniversal<ProblemShape, BlockMmad, BlockEpilogue, BlockScheduler>;

    using MatmulAdapter = Gemm::Device::DeviceGemm<MatmulKernel>;

    using Arguments = typename MatmulKernel::Arguments;
    Arguments arguments = {
        shape,
        {deviceA, deviceB, deviceC},
        {deviceC,
         x2ScaleDevice,
         x1ScaleDevice,
         quantBiasDevice,
         {tla::get<0>(L1TileShape{}), tla::get<1>(L1TileShape{}), x1QuantMode, x2QuantMode, AscendC::DT_FLOAT,
          quantBiasDevice != nullptr}}};

    MatmulAdapter matmulOp;
    matmulOp.CanImplement(arguments);
    sizeWorkspace = matmulOp.GetWorkspaceSize(arguments);
    if (sizeWorkspace > 0) {
        ACL_CHECK(aclrtMalloc(reinterpret_cast<void**>(&deviceWorkspace), sizeWorkspace, ACL_MEM_MALLOC_HUGE_FIRST));
    }
    matmulOp.Initialize(arguments, deviceWorkspace);
    matmulOp(stream, aicCoreUsed);

    ACL_CHECK(aclrtSynchronizeStream(stream));

    std::vector<fp16_t> hostC(lenC);
    ACL_CHECK(aclrtMemcpy(hostC.data(), sizeC, deviceC, sizeC, ACL_MEMCPY_DEVICE_TO_HOST));

    std::vector<float> hostGolden(lenC);
    std::string expected_path = "./output/golden_o.bin";
    if (!ReadFile(expected_path, hostGolden.data(), goldenSize)) {
        cleanupAndFinalize();
        return;
    }

    std::vector<uint64_t> errorIndices = golden::CompareData(hostC, hostGolden, k);
    if (errorIndices.empty()) {
        std::cout << "Compare success." << std::endl;
    } else {
        std::cerr << "Compare failed. Error count: " << errorIndices.size() << std::endl;
    }

    cleanupAndFinalize();
}

QuantMode StringToQuantMode(const std::string& s)
{
    if (s == "per_tensor") {
        return QuantMode::PERTENSOR_MODE;
    } else if (s == "per_channel") {
        return QuantMode::PERCHANNEL_MODE;
    } else if (s == "per_token") {
        return QuantMode::PERTOKEN_MODE;
    } else if (s == "default") {
        return QuantMode::DEFAULT;
    } else {
        throw std::invalid_argument("Invalid quant mode string: " + s);
    }
}

struct ArgsParams {
    uint32_t m;
    uint32_t n;
    uint32_t k;
    QuantMode x1QuantMode;
    QuantMode x2QuantMode;
    bool enableBias = false;
};

bool IsValidQuantMode(QuantMode x1QuantMode, QuantMode x2QuantMode, bool enableBias)
{
    using TupleType = std::tuple<QuantMode, QuantMode, bool>;
    static const std::vector<TupleType> validPairs = {
        {QuantMode::PERTOKEN_MODE, QuantMode::PERTENSOR_MODE, false},
        {QuantMode::PERTOKEN_MODE, QuantMode::PERCHANNEL_MODE, false},
        {QuantMode::PERTENSOR_MODE, QuantMode::PERCHANNEL_MODE, false},
        {QuantMode::DEFAULT, QuantMode::PERCHANNEL_MODE, false},
        {QuantMode::PERTOKEN_MODE, QuantMode::PERTENSOR_MODE, true},
        {QuantMode::PERTOKEN_MODE, QuantMode::PERCHANNEL_MODE, true},
        {QuantMode::DEFAULT, QuantMode::PERTENSOR_MODE, true},
        {QuantMode::DEFAULT, QuantMode::PERCHANNEL_MODE, true}};

    const auto target = std::make_tuple(x1QuantMode, x2QuantMode, enableBias);
    return std::find(validPairs.begin(), validPairs.end(), target) != validPairs.end();
}

ArgsParams ParseArguments(int32_t argc, const char* argv[])
{
    ArgsParams params;
    constexpr static int32_t minArgs = 5;
    if (argc < minArgs) {
        throw std::invalid_argument("Insufficient arguments provided");
    }

    try {
        params.m = std::stoi(argv[1]);
        params.n = std::stoi(argv[2]);
        params.k = std::stoi(argv[3]);

        params.x1QuantMode = StringToQuantMode(argv[4]);
        params.x2QuantMode = StringToQuantMode(argv[5]);

        if (argc >= 7) {
            if (std::string(argv[6]) == "has_bias") {
                params.enableBias = true;
            } else {
                throw std::invalid_argument("Invalid argument for bias flag: " + std::string(argv[6]));
            }
        }
    } catch (const std::exception& e) {
        throw std::invalid_argument(std::string("Error parsing arguments: ") + e.what());
    }
    return params;
}

int main(int argc, const char** argv)
{
    for (int32_t i = 0; i < argc; i++) {
        std::cout << "arg[" << i << "]: " << argv[i] << std::endl;
    }

    ArgsParams params = ParseArguments(argc, argv);
    std::cout << "m: " << params.m << std::endl
              << "n: " << params.n << std::endl
              << "k: " << params.k << std::endl
              << "x1QuantMode: " << static_cast<int>(params.x1QuantMode) << std::endl
              << "x2QuantMode: " << static_cast<int>(params.x2QuantMode) << std::endl
              << "enableBias: " << (params.enableBias ? "true" : "false") << std::endl;

    if (!IsValidQuantMode(params.x1QuantMode, params.x2QuantMode, params.enableBias)) {
        std::cerr << "Invalid combination of quantization modes and bias flag." << std::endl;
        return -1;
    }
    Options options;
    options.problemShape = {params.m, params.n, params.k};
    options.deviceId = 0;
    Run(options, params.x1QuantMode, params.x2QuantMode, params.enableBias);
    return 0;
}
