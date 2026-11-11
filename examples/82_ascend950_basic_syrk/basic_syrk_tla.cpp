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
#include "catlass/gemm/kernel/basic_syrk_tla.hpp"

#include "catlass/arch/arch.hpp"
#include "catlass/catlass.hpp"
#include "catlass/gemm/block/block_mmad_syrk_tla.hpp"
#include "catlass/gemm/block/block_scheduler_syrk.h"
#include "catlass/gemm/device/device_gemm.hpp"
#include "catlass/layout/layout.hpp"
#include "catlass/status.hpp"
#include "tla/layout.hpp"

#include "golden.hpp"
#include "helper.hpp"

using namespace Catlass;
using namespace tla;

using Options = SyrkOptions;

static int Run(const Options& options)
{
    using ElementX = bfloat16_t;
    using ElementY = bfloat16_t;
    using L1TileShape = Shape<Int<256>, Int<256>, Int<128>>;
    using L0TileShape = Shape<Int<256>, Int<256>, Int<64>>;
    // Layout tags are fixed by BlockMmadSyrkTla; use the default TileCopy.
    using BlockMmad = Gemm::Block::BlockMmadSyrkTla<L1TileShape, L0TileShape, ElementX, ElementY>;
    using BlockEpilogue = void;
    using BlockScheduler = Gemm::Block::SyrkBlockScheduler;
    using MatmulKernel = Gemm::Kernel::BasicSyrkTla<BlockMmad, BlockEpilogue, BlockScheduler>;
    using MatmulAdapter = Gemm::Device::DeviceGemm<MatmulKernel>;

    MatmulKernel::Arguments arguments{1, options.problemShape, nullptr, nullptr, nullptr};
    if (MatmulAdapter::CanImplement(arguments) != Status::kSuccess) {
        std::cerr << "Unsupported SYRK shape: require M=N, 1<=M<=8192, 1<=K<=65536 and M*K<=67108864." << std::endl;
        return 1;
    }

    aclrtStream stream{nullptr};

    ACL_CHECK(aclInit(nullptr));
    ACL_CHECK(aclrtSetDevice(options.deviceId));
    ACL_CHECK(aclrtCreateStream(&stream));

    // Y = X * X^T, X: [M, K], Y: [M, M]
    uint32_t m = options.problemShape.m();
    uint32_t k = options.problemShape.k();

    // Host-side tags for golden only; device layouts are fixed inside BlockMmadSyrkTla.
    using LayoutTagX = layout::RowMajor;
    using LayoutTagXt = layout::ColumnMajor;
    using LayoutTagY = layout::RowMajor;

    LayoutTagX tagX = LayoutTagX::MakeLayout<ElementX>(m, k);
    LayoutTagXt tagXt = LayoutTagXt::MakeLayout<ElementX>(k, m);
    LayoutTagY tagY = LayoutTagY::MakeLayout<ElementY>(m, m);

    size_t lenX = tagX.Capacity();
    size_t lenY = tagY.Capacity();

    size_t sizeX = lenX * sizeof(ElementX);
    size_t sizeY = lenY * sizeof(ElementY);

    std::vector<bfloat16> hostX(lenX);
    golden::FillRandomData<bfloat16>(hostX, -5.0f, 5.0f);

    uint8_t* deviceX{nullptr};
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void**>(&deviceX), sizeX, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemcpy(deviceX, sizeX, hostX.data(), sizeX, ACL_MEMCPY_HOST_TO_DEVICE));

    uint8_t* deviceY{nullptr};
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void**>(&deviceY), sizeY, ACL_MEM_MALLOC_HUGE_FIRST));

    uint8_t* deviceWorkspace{nullptr};

    auto aicCoreNum = platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreNumAic();

    BlockScheduler scheduler(
        options.problemShape, Catlass::MakeCoord(MatmulKernel::L1_TILE_M, MatmulKernel::L1_TILE_N));
    uint32_t taskNum = scheduler.GetCoreLoops();
    uint32_t aicCoreUsed = min(aicCoreNum, taskNum);

    arguments.ptrX = deviceX;
    arguments.ptrXt = deviceX;
    arguments.ptrY = deviceY;

    MatmulAdapter matmulOp;
    size_t sizeWorkspace = matmulOp.GetWorkspaceSize(arguments);
    if (sizeWorkspace > 0) {
        ACL_CHECK(aclrtMalloc(reinterpret_cast<void**>(&deviceWorkspace), sizeWorkspace, ACL_MEM_MALLOC_HUGE_FIRST));
    }
    matmulOp.Initialize(arguments, deviceWorkspace);
    matmulOp(stream, aicCoreUsed);
    ACL_CHECK(aclrtSynchronizeStream(stream));

    std::vector<bfloat16> hostY(lenY);
    ACL_CHECK(aclrtMemcpy(hostY.data(), sizeY, deviceY, sizeY, ACL_MEMCPY_DEVICE_TO_HOST));

    std::vector<float> hostGolden;
    {
        // ComputeMatmul accumulates in its output element type. Use fp64 to
        // avoid long-K cancellation errors, then round once for CompareData.
        std::vector<double> hostGolden64(lenY);
        golden::ComputeMatmul(options.problemShape, hostX, tagX, hostX, tagXt, hostGolden64, tagY);
        hostGolden.assign(hostGolden64.begin(), hostGolden64.end());
    }

    std::vector<uint64_t> errorIndices = golden::CompareData(hostY, hostGolden, k);
    if (errorIndices.empty()) {
        std::cout << "Compare success." << std::endl;
    } else {
        std::cerr << "Compare failed. Error count: " << errorIndices.size() << std::endl;
    }

    ACL_CHECK(aclrtFree(deviceX));
    ACL_CHECK(aclrtFree(deviceY));
    if (sizeWorkspace > 0) {
        ACL_CHECK(aclrtFree(deviceWorkspace));
    }

    ACL_CHECK(aclrtDestroyStream(stream));
    ACL_CHECK(aclrtResetDevice(options.deviceId));
    ACL_CHECK(aclFinalize());
    return errorIndices.empty() ? 0 : 1;
}

int main(int argc, const char** argv)
{
    // This example implements only Y = X * X^T, without batch/alpha/beta.
    if (argc != 3 && argc != 4) {
        std::cerr << "Usage: " << argv[0] << " m k [device_id]" << std::endl;
        return 1;
    }
    Options options;
    if (options.Parse(argc, argv) != 0) {
        return -1;
    }
    return Run(options);
}
