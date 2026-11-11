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

#include "catlass/gemm/kernel/matrix_inverse.hpp"

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
#include "tla/tensor.hpp"

#include <cmath>
#include <cstdlib>

#include "golden.hpp"
#include "helper.hpp"

using namespace Catlass;
using namespace tla;

static void Run(uint32_t N, int32_t deviceId)
{
    using Element = float;
    using LayoutTag = layout::RowMajor;

    // Define kernel types
    using ArchTag = Arch::AtlasA2;
    constexpr bool enableUnitFlag = true;
    constexpr bool useHF32 = false;

    using DispatchPolicy = Gemm::MmadPingpong<ArchTag, enableUnitFlag, useHF32>;
    using L1TileShape = Shape<_128, _128, _256>;
    using L0TileShape = Shape<_128, _128, _64>;
    using TileCopy = Gemm::Tile::PackedTileCopyTla<ArchTag, Element, LayoutTag, Element, LayoutTag, Element, LayoutTag>;
    using BlockMmadType =
        Gemm::Block::BlockMmadTla<DispatchPolicy, L1TileShape, L0TileShape, Element, Element, Element, void, TileCopy>;
    using BlockSchedulerType = Gemm::Block::GemmIdentityBlockSwizzle<>;

    using InverterKernel = Gemm::Kernel::MatrixInverse<ArchTag, float, BlockMmadType, BlockSchedulerType>;
    using InverterAdapter = Gemm::Device::DeviceGemm<InverterKernel>;

    // CanImplement only depends on N, so validate it before initializing the device.
    auto layoutA = tla::MakeLayout<Element, LayoutTag>(N, N);
    InverterKernel::Arguments probeArgs{N, nullptr, layoutA, nullptr, nullptr};
    if (InverterAdapter::CanImplement(probeArgs) != Status::kSuccess) {
        std::cerr << "Unsupported N=" << N << ": matrix inverse requires N <= "
                  << ArchTag::UB_SIZE / (3 * sizeof(Element)) << "." << std::endl;
        return;
    }

    aclrtStream stream{nullptr};

    ACL_CHECK(aclInit(nullptr));
    ACL_CHECK(aclrtSetDevice(deviceId));
    ACL_CHECK(aclrtCreateStream(&stream));

    size_t lenA = static_cast<size_t>(N) * N;
    size_t lenIpiv = static_cast<size_t>(N);
    size_t sizeA = lenA * sizeof(Element);
    size_t sizeIpiv = lenIpiv * sizeof(int32_t);

    // Generate test matrix
    std::vector<float> hostA(lenA);
    // Generate a diagonally dominant random matrix (well-conditioned)

    golden::FillRandomData(hostA, -1.0f, 1.0f);
    for (uint32_t i = 0; i < N; ++i) {
        hostA[i * N + i] += static_cast<float>(N); // strengthen diagonal
    }

    // Keep a copy for golden comparison
    std::vector<float> hostOriginal = hostA;

    // Allocate device memory
    uint8_t* deviceA{nullptr};
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void**>(&deviceA), sizeA, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemcpy(deviceA, sizeA, hostA.data(), sizeA, ACL_MEMCPY_HOST_TO_DEVICE));

    uint8_t* deviceIpiv{nullptr};
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void**>(&deviceIpiv), sizeIpiv, ACL_MEM_MALLOC_HUGE_FIRST));

    // Get core count
    auto aicCoreNum = platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreNumAic();

    // Set up arguments
    InverterKernel::Arguments arguments{N, deviceA, layoutA, deviceIpiv, nullptr};

    InverterAdapter invOp;

    size_t sizeWorkspace = invOp.GetWorkspaceSize(arguments);
    uint8_t* deviceWorkspace = nullptr;
    if (sizeWorkspace > 0) {
        ACL_CHECK(aclrtMalloc(reinterpret_cast<void**>(&deviceWorkspace), sizeWorkspace, ACL_MEM_MALLOC_HUGE_FIRST));
        // Re-set arguments with workspace pointer
        arguments = InverterKernel::Arguments{N, deviceA, layoutA, deviceIpiv, deviceWorkspace};
    }

    invOp.Initialize(arguments, deviceWorkspace);

    // Kernel timing with ACL events
    aclrtEvent startEvent, endEvent;
    ACL_CHECK(aclrtCreateEvent(&startEvent));
    ACL_CHECK(aclrtCreateEvent(&endEvent));
    ACL_CHECK(aclrtRecordEvent(startEvent, stream));
    invOp(stream, aicCoreNum);
    ACL_CHECK(aclrtRecordEvent(endEvent, stream));
    ACL_CHECK(aclrtSynchronizeEvent(endEvent));

    float kernelTimeMs = 0.0f;
    ACL_CHECK(aclrtEventElapsedTime(&kernelTimeMs, startEvent, endEvent));
    aclrtDestroyEvent(startEvent);
    aclrtDestroyEvent(endEvent);
    std::cout << "Kernel time: " << kernelTimeMs << " ms" << std::endl;

    if (sizeWorkspace > 0) {
        ACL_CHECK(aclrtFree(deviceWorkspace));
    }

    // Copy result back
    std::vector<float> hostResult(lenA);
    ACL_CHECK(aclrtMemcpy(hostResult.data(), sizeA, deviceA, sizeA, ACL_MEMCPY_DEVICE_TO_HOST));

    // Golden reference
    std::vector<float> hostGolden = hostOriginal;
    int info = golden::ComputeInverseInplace(N, hostGolden);
    if (info != 0) {
        std::cerr << "Golden reference failed: matrix is singular (info=" << info << ")" << std::endl;
    }

    // Compare
    std::vector<uint64_t> errorIndices = golden::CompareData(hostResult, hostGolden, N * N);
    if (errorIndices.empty()) {
        std::cout << "Compare success." << std::endl;
    } else {
        std::cerr << "Compare failed. Error count: " << errorIndices.size() << " / " << lenA << std::endl;
    }

    ACL_CHECK(aclrtFree(deviceA));
    ACL_CHECK(aclrtFree(deviceIpiv));
    ACL_CHECK(aclrtDestroyStream(stream));
    ACL_CHECK(aclrtResetDevice(deviceId));
    ACL_CHECK(aclFinalize());
}

int main(int argc, const char** argv)
{
    uint32_t N = 64;
    int32_t deviceId = 0;

    if (argc >= 2) {
        N = static_cast<uint32_t>(std::atoi(argv[1]));
    }
    if (argc >= 3) {
        deviceId = std::atoi(argv[2]);
    }

    std::cout << "Matrix Inverse: N=" << N << ", device=" << deviceId << std::endl;
    Run(N, deviceId);
    return 0;
}
