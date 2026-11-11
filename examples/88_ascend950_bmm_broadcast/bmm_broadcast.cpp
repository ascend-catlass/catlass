/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef K_MAX_SHAPE_DIM
#define K_MAX_SHAPE_DIM 0
#endif

#include <iostream>
#include "catlass/gemm/kernel/batched_matmul_broadcast_tla.h"

#include "catlass/gemm/block/block_mmad.hpp"
#include "catlass/gemm/block/block_swizzle.hpp"
#include "catlass/gemm/device/device_gemm.hpp"
#include "catlass/gemm/tile/tile_copy.hpp"

#include "golden.hpp"
#include "helper.hpp"

using namespace Catlass;
using namespace tla;

static bool Run(const BatchedMatmulBroadcastTlaOptions& options, aclrtStream stream)
{
    using Element = half;
    using HostElement = fp16_t;
    using ElementBias = half;

    using LayoutTagA = layout::RowMajor;
    using LayoutTagB = layout::RowMajor;
    constexpr bool useHF32 = false;

    // Default stages: single L0C, double L1A/L1B/L0A/L0B, no L1 residency.
    using DispatchPolicy = Gemm::MmadPingpong<Arch::Ascend950, false, useHF32>;
    using L1TileShape = tla::Shape<tla::Int<64>, tla::Int<128>, tla::Int<128>>;
    using L0TileShape = tla::Shape<tla::Int<64>, tla::Int<128>, tla::Int<32>>;
    using BaseTileCopy = Gemm::Tile::PackedTileCopyTla<
        Arch::Ascend950, Element, LayoutTagA, Element, LayoutTagB, Element, layout::RowMajor, ElementBias>;
    using TileCopy = Gemm::Tile::PaddedTileCopy<BaseTileCopy>;
    using BlockMmad = Gemm::Block::BlockMmadTla<
        DispatchPolicy, L1TileShape, L0TileShape, Element, Element, Element, ElementBias, TileCopy>;

    // Swizzle offset 3 groups three M tiles; direction 0 selects the Zn traversal.
    using MatmulKernel =
        Gemm::Kernel::BatchedMatmulBroadcastTla<BlockMmad, Gemm::Block::GemmIdentityBlockSwizzle<3, 0>>;
    using MatmulAdapter = Gemm::Device::DeviceGemm<MatmulKernel>;

    using BiasMode = Gemm::Kernel::BiasMode;
    constexpr BiasMode biasMode = BiasMode::SharedN;

    auto layoutA = tla::MakeLayout<Element, LayoutTagA>(options.problemShape.m(), options.problemShape.k());
    auto layoutB = tla::MakeLayout<Element, LayoutTagB>(options.problemShape.k(), options.problemShape.n());
    auto layoutC = tla::MakeLayout<Element, layout::RowMajor>(options.problemShape.m(), options.problemShape.n());

    // Validate dimensions before allocating device memory; this host address is never dereferenced.
    alignas(sizeof(Element)) uint8_t placeholderStorage{};
    auto* placeholder = &placeholderStorage;
    MatmulKernel::Arguments arguments{
        options.batchCount, options.batchA,
        options.batchB,     GemmCoord{options.problemShape.m(), options.problemShape.n(), options.problemShape.k()},
        placeholder,        layoutA,
        placeholder,        layoutB,
        placeholder,        layoutC,
        placeholder,        biasMode};
    if (!MatmulKernel::CanImplement(arguments)) {
        throw std::invalid_argument("Unsupported BMM broadcast arguments");
    }

    size_t lenA = static_cast<size_t>(options.batchA) * options.problemShape.m() * options.problemShape.k();
    size_t lenB = static_cast<size_t>(options.batchB) * options.problemShape.k() * options.problemShape.n();
    size_t lenC = static_cast<size_t>(options.batchCount) * options.problemShape.m() * options.problemShape.n();
    size_t lenBias = options.problemShape.n();

    constexpr size_t guardElements = 32;
    const HostElement guardValue = static_cast<HostElement>(-113.0f);

    std::vector<HostElement> hostA(lenA);
    std::vector<HostElement> hostB(lenB);
    std::vector<HostElement> hostBias(lenBias);
    std::vector<HostElement> hostC(lenC + 2 * guardElements, guardValue);

    // Deterministic binary fractions are exactly representable in all supported input formats.
    for (size_t i = 0; i < lenA; ++i) {
        hostA[i] = static_cast<HostElement>((static_cast<int>(i * 17 % 29) - 14) / 16.0f);
    }

    for (size_t i = 0; i < lenB; ++i) {
        hostB[i] = static_cast<HostElement>((static_cast<int>(i * 11 % 31) - 15) / 16.0f);
    }

    for (size_t i = 0; i < lenBias; ++i) {
        hostBias[i] = static_cast<HostElement>((static_cast<int>(i * 7 % 23) - 11) / 8.0f);
    }

    uint8_t* deviceA = nullptr;
    uint8_t* deviceB = nullptr;
    uint8_t* deviceC = nullptr;
    uint8_t* deviceBias = nullptr;
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void**>(&deviceA), lenA * sizeof(Element), ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void**>(&deviceB), lenB * sizeof(Element), ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(
        aclrtMalloc(reinterpret_cast<void**>(&deviceC), hostC.size() * sizeof(Element), ACL_MEM_MALLOC_HUGE_FIRST));

    ACL_CHECK(
        aclrtMemcpy(deviceA, lenA * sizeof(Element), hostA.data(), lenA * sizeof(Element), ACL_MEMCPY_HOST_TO_DEVICE));
    ACL_CHECK(
        aclrtMemcpy(deviceB, lenB * sizeof(Element), hostB.data(), lenB * sizeof(Element), ACL_MEMCPY_HOST_TO_DEVICE));
    ACL_CHECK(aclrtMemcpy(
        deviceC, hostC.size() * sizeof(Element), hostC.data(), hostC.size() * sizeof(Element),
        ACL_MEMCPY_HOST_TO_DEVICE));

    ACL_CHECK(aclrtMalloc(reinterpret_cast<void**>(&deviceBias), lenBias * sizeof(Element), ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemcpy(
        deviceBias, lenBias * sizeof(Element), hostBias.data(), lenBias * sizeof(Element), ACL_MEMCPY_HOST_TO_DEVICE));

    arguments.ptrA = deviceA;
    arguments.ptrB = deviceB;
    arguments.ptrC = deviceC + guardElements * sizeof(Element);
    arguments.ptrBias = deviceBias;

    MatmulAdapter matmulAdapter;
    if (matmulAdapter.CanImplement(arguments) != Status::kSuccess) {
        throw std::runtime_error("Unsupported BMM broadcast arguments");
    }
    matmulAdapter.Initialize(arguments);
    auto aicCoreNum = platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreNumAic();
    if (matmulAdapter(stream, aicCoreNum) != Status::kSuccess) {
        throw std::runtime_error("BMM broadcast launch failed");
    }

    ACL_CHECK(aclrtSynchronizeStream(stream));
    ACL_CHECK(aclrtMemcpy(
        hostC.data(), hostC.size() * sizeof(Element), deviceC, hostC.size() * sizeof(Element),
        ACL_MEMCPY_DEVICE_TO_HOST));

    std::vector<float> hostGolden(lenC);
    golden::ComputeBatchedMatmulBroadcast<HostElement, LayoutTagA, LayoutTagB>(
        options.problemShape, options.batchA, options.batchB, static_cast<uint32_t>(biasMode), hostA, hostB, hostBias,
        hostGolden);
    size_t errors = golden::CompareBatchedMatmulBroadcast<HostElement, false, useHF32>(
        hostC, hostGolden, guardElements, guardValue);

    ACL_CHECK(aclrtFree(deviceA));
    ACL_CHECK(aclrtFree(deviceB));
    ACL_CHECK(aclrtFree(deviceC));
    ACL_CHECK(aclrtFree(deviceBias));

    if (errors == 0) {
        std::cout << "Compare success." << std::endl;
    } else {
        std::cout << "Compare failed. Error count: " << errors << std::endl;
    }
    return errors == 0;
}

int main(int argc, const char** argv)
{
    BatchedMatmulBroadcastTlaOptions options;
    if (options.Parse(argc, argv) != 0) {
        return -1;
    }

    ACL_CHECK(aclInit(nullptr));
    ACL_CHECK(aclrtSetDevice(options.deviceId));
    aclrtStream stream = nullptr;
    ACL_CHECK(aclrtCreateStream(&stream));

    bool success = false;
    try {
        success = Run(options, stream);
    } catch (const std::exception& error) {
        std::cerr << error.what() << std::endl;
    }

    ACL_CHECK(aclrtDestroyStream(stream));
    ACL_CHECK(aclrtResetDevice(options.deviceId));
    ACL_CHECK(aclFinalize());

    if (!success) {
        return -1;
    }
    return 0;
}
