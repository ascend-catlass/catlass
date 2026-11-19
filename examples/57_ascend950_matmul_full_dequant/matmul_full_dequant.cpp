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

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <type_traits>
#include <vector>

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

using QuantMode = Epilogue::QuantMode;

QuantMode StringToQuantMode(const std::string& s)
{
    if (s == "per_tensor") {
        return QuantMode::PERTENSOR_MODE;
    } else if (s == "per_channel") {
        return QuantMode::PERCHANNEL_MODE;
    } else if (s == "per_token") {
        return QuantMode::PERTOKEN_MODE;
    } else if (s == "none" || s == "default") {
        return QuantMode::NONE;
    } else {
        throw std::invalid_argument("Invalid quant mode string: " + s);
    }
}

struct DequantGemmOptions : public GemmOptions {
    const std::string HELPER = "m n k x1_quant_mode x2_quant_mode [has_bias]";

    QuantMode x1QuantMode{QuantMode::PERTOKEN_MODE};
    QuantMode x2QuantMode{QuantMode::PERCHANNEL_MODE};
    bool enableBias{false};

    DequantGemmOptions() = default;

    int Parse(int argc, const char** argv)
    {
        constexpr static int32_t minArgs = 5;
        if (argc < minArgs) {
            std::cerr << TOSTRING(CATLASS_EXAMPLE_NAME) << " " << HELPER << std::endl;
            return -1;
        }

        try {
            problemShape.m() = std::stoi(argv[1]);
            problemShape.n() = std::stoi(argv[2]);
            problemShape.k() = std::stoi(argv[3]);
            if (problemShape.m() == 0 || problemShape.n() == 0 || problemShape.k() == 0) {
                std::cerr << TOSTRING(CATLASS_EXAMPLE_NAME) << ": m, n and k must be greater than 0" << std::endl;
                return -1;
            }

            x1QuantMode = StringToQuantMode(argv[4]);
            if (argc == minArgs) {
                std::cerr << "Missing x2 quant mode" << std::endl;
                return -1;
            }
            x2QuantMode = StringToQuantMode(argv[5]);

            if (argc >= 7) {
                if (std::string(argv[6]) == "has_bias") {
                    enableBias = true;
                } else {
                    std::cerr << "Invalid argument for bias flag: " << argv[6] << std::endl;
                    return -1;
                }
            }
        } catch (const std::exception& e) {
            std::cerr << "Error parsing arguments: " << e.what() << std::endl;
            return -1;
        }
        return 0;
    }

    const bool CanImplement() const
    {
        using TupleType = std::tuple<QuantMode, QuantMode, bool>;
        static const std::vector<TupleType> validPairs = {
            {QuantMode::PERTOKEN_MODE, QuantMode::PERTENSOR_MODE, false},
            {QuantMode::PERTOKEN_MODE, QuantMode::PERCHANNEL_MODE, false},
            {QuantMode::PERTENSOR_MODE, QuantMode::PERCHANNEL_MODE, false},
            {QuantMode::NONE, QuantMode::PERCHANNEL_MODE, false},
            {QuantMode::PERTOKEN_MODE, QuantMode::PERTENSOR_MODE, true},
            {QuantMode::PERTOKEN_MODE, QuantMode::PERCHANNEL_MODE, true},
            {QuantMode::NONE, QuantMode::PERTENSOR_MODE, true},
            {QuantMode::NONE, QuantMode::PERCHANNEL_MODE, true}};

        const auto target = std::make_tuple(x1QuantMode, x2QuantMode, enableBias);
        return std::find(validPairs.begin(), validPairs.end(), target) != validPairs.end();
    }
};

using Options = DequantGemmOptions;

template <QuantMode X1_QUANT_MODE, QuantMode X2_QUANT_MODE, class ElementBias>
static void Run(const Options& options)
{
    constexpr bool hasBias = !std::is_void_v<ElementBias>;
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
    using ElementX1Scale = float;
    using ElementX2Scale = float;

    using LayoutTagA = layout::RowMajor;
    using LayoutTagB = layout::RowMajor;
    using LayoutTagC = layout::RowMajor;
    LayoutTagA tagA = LayoutTagA::MakeLayout<ElementA>(m, k);
    LayoutTagB tagB = LayoutTagB::MakeLayout<ElementB>(k, n);
    LayoutTagC tagC = LayoutTagC::MakeLayout<ElementC>(m, n);

    size_t lenA = tagA.Capacity();
    size_t lenB = tagB.Capacity();
    size_t lenC = tagC.Capacity();
    size_t sizeA = lenA * sizeof(ElementA);
    size_t sizeB = lenB * sizeof(ElementB);
    size_t sizeC = lenC * sizeof(ElementD);
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
        if (deviceWorkspace)
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
    if constexpr (X1_QUANT_MODE == QuantMode::PERTENSOR_MODE) {
        x1ScaleFileSize = sizeof(float);
    } else if constexpr (X1_QUANT_MODE == QuantMode::PERTOKEN_MODE) {
        x1ScaleFileSize = m * sizeof(float);
    }
    if constexpr (X1_QUANT_MODE != QuantMode::NONE) {
        ACL_CHECK(aclrtMallocHost((void**)(&x1ScaleHost), x1ScaleFileSize));
        ACL_CHECK(aclrtMalloc((void**)&x1ScaleDevice, x1ScaleFileSize, ACL_MEM_MALLOC_HUGE_FIRST));
        if (!ReadFile("./input/x1_scale.bin", x1ScaleHost, x1ScaleFileSize)) {
            cleanupAndFinalize();
            return;
        }
        ACL_CHECK(aclrtMemcpy(x1ScaleDevice, x1ScaleFileSize, x1ScaleHost, x1ScaleFileSize, ACL_MEMCPY_HOST_TO_DEVICE));
    }

    size_t x2ScaleFileSize = 0;
    if constexpr (X2_QUANT_MODE == QuantMode::PERTENSOR_MODE) {
        x2ScaleFileSize = sizeof(float);
    } else if constexpr (X2_QUANT_MODE == QuantMode::PERCHANNEL_MODE) {
        x2ScaleFileSize = n * sizeof(float);
    }
    ACL_CHECK(aclrtMallocHost((void**)(&x2ScaleHost), x2ScaleFileSize));
    ACL_CHECK(aclrtMalloc((void**)&x2ScaleDevice, x2ScaleFileSize, ACL_MEM_MALLOC_HUGE_FIRST));
    if (!ReadFile("./input/x2_scale.bin", x2ScaleHost, x2ScaleFileSize)) {
        cleanupAndFinalize();
        return;
    }
    ACL_CHECK(aclrtMemcpy(x2ScaleDevice, x2ScaleFileSize, x2ScaleHost, x2ScaleFileSize, ACL_MEMCPY_HOST_TO_DEVICE));

    if constexpr (hasBias) {
        size_t quantBiasFileSize = n * sizeof(ElementBias);
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

#if defined(CATLASS_ARCH) && CATLASS_ARCH == 9201
    using ArchTag = Arch::Ascend960DT;
#elif defined(CATLASS_ARCH) && CATLASS_ARCH == 9202
    using ArchTag = Arch::Ascend960PR;
#else
    using ArchTag = Arch::Ascend950;
#endif
    constexpr bool enableUnitFlag = true;
    using DispatchPolicy = Gemm::MmadPingpong<ArchTag, enableUnitFlag>;

    using L1TileShape = Shape<Int<128>, Int<256>, Int<256>>;
    using L0TileShape = Shape<Int<128>, Int<256>, Int<128>>;

    constexpr uint32_t taskRatio = 2;
    using TileCopy = Gemm::Tile::PackedTileCopyTlaToUB<
        ArchTag, ElementA, LayoutTagA, ElementB, LayoutTagB, ElementC, LayoutTagC, ElementBias,
#if defined(CATLASS_ARCH) && (CATLASS_ARCH == 9201 || CATLASS_ARCH == 9202)
        Gemm::Tile::CopyL0CToUBMode::NO_SPLIT>;
#else
        Gemm::Tile::CopyL0CToUBMode::SPLIT_M>;
#endif
    using BlockMmad = Gemm::Block::BlockMmadTla<
        DispatchPolicy, L1TileShape, L0TileShape, ElementA, ElementB, ElementC, ElementBias, TileCopy>;
    using EpilogueDispatchPolicy = Epilogue::BlockEpilogueDequant<X1_QUANT_MODE, X2_QUANT_MODE>;
    using BlockEpilogue = Epilogue::Block::BlockEpilogue<
        EpilogueDispatchPolicy, L0TileShape, ElementC, ElementD, ElementBias, ElementX1Scale, ElementX2Scale,
        std::integral_constant<uint32_t, taskRatio>>;

    uint32_t taskNum = CeilDiv(options.problemShape.m(), tla::get<0>(L1TileShape{})) *
                       CeilDiv(options.problemShape.n(), tla::get<1>(L1TileShape{}));
    uint32_t aicCoreUsed = std::min(aicCoreNum, taskNum);

    using BlockScheduler = typename Gemm::Block::BlockSchedulerAswt<L1TileShape, L0TileShape>;

    // kernel level
    using MatmulKernel =
        Gemm::Kernel::GemmUniversal<BlockMmad, BlockEpilogue, BlockScheduler, LayoutTagA, LayoutTagB, LayoutTagC>;

    using MatmulAdapter = Gemm::Device::DeviceGemm<MatmulKernel>;

    using Arguments = typename MatmulKernel::Arguments;
    Arguments arguments = {options.problemShape, deviceA,       deviceB,        deviceC,
                           x2ScaleDevice,        x1ScaleDevice, quantBiasDevice};

    MatmulAdapter matmulOp;
    if (matmulOp.CanImplement(arguments) != Status::kSuccess) {
        std::cerr << "The arguments are not supported by the kernel." << std::endl;
        cleanupAndFinalize();
        return;
    }
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

void DispatchRun(const Options& options)
{
    const QuantMode x1QuantMode = options.x1QuantMode;
    const QuantMode x2QuantMode = options.x2QuantMode;
    const bool enableBias = options.enableBias;
    if (x1QuantMode == QuantMode::PERTOKEN_MODE && x2QuantMode == QuantMode::PERTENSOR_MODE) {
        if (enableBias) {
            Run<QuantMode::PERTOKEN_MODE, QuantMode::PERTENSOR_MODE, float>(options);
        } else {
            Run<QuantMode::PERTOKEN_MODE, QuantMode::PERTENSOR_MODE, void>(options);
        }
    } else if (x1QuantMode == QuantMode::PERTOKEN_MODE && x2QuantMode == QuantMode::PERCHANNEL_MODE) {
        if (enableBias) {
            Run<QuantMode::PERTOKEN_MODE, QuantMode::PERCHANNEL_MODE, float>(options);
        } else {
            Run<QuantMode::PERTOKEN_MODE, QuantMode::PERCHANNEL_MODE, void>(options);
        }
    } else if (x1QuantMode == QuantMode::PERTENSOR_MODE && x2QuantMode == QuantMode::PERCHANNEL_MODE) {
        Run<QuantMode::PERTENSOR_MODE, QuantMode::PERCHANNEL_MODE, void>(options);
    } else if (x1QuantMode == QuantMode::NONE && x2QuantMode == QuantMode::PERTENSOR_MODE) {
        Run<QuantMode::NONE, QuantMode::PERTENSOR_MODE, float>(options);
    } else if (x1QuantMode == QuantMode::NONE && x2QuantMode == QuantMode::PERCHANNEL_MODE) {
        if (enableBias) {
            Run<QuantMode::NONE, QuantMode::PERCHANNEL_MODE, float>(options);
        } else {
            Run<QuantMode::NONE, QuantMode::PERCHANNEL_MODE, void>(options);
        }
    }
}

int main(int argc, const char** argv)
{
    Options options;
    if (options.Parse(argc, argv) != 0) {
        return -1;
    }
    std::cout << "m: " << options.problemShape.m() << std::endl
              << "n: " << options.problemShape.n() << std::endl
              << "k: " << options.problemShape.k() << std::endl
              << "x1QuantMode: " << static_cast<int>(options.x1QuantMode) << std::endl
              << "x2QuantMode: " << static_cast<int>(options.x2QuantMode) << std::endl
              << "enableBias: " << (options.enableBias ? "true" : "false") << std::endl;

    if (!options.CanImplement()) {
        std::cerr << "Invalid combination of quantization modes and bias flag." << std::endl;
        return -1;
    }
    DispatchRun(options);
    return 0;
}
