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

#include "catlass/arch/arch.hpp"
#include "catlass/catlass.hpp"
#include "catlass/epilogue/block/block_epilogue_syrk_axpby.hpp"
#include "catlass/gemm/block/block_mmad_syrk_tla.hpp"
#include "catlass/gemm/block/block_scheduler_syrk.h"
#include "catlass/gemm/kernel/basic_syrk_workspace_tla.hpp"
#include "catlass/gemm_coord.hpp"
#include "catlass/layout/layout.hpp"
#include "tla/layout.hpp"

#include "golden.hpp"
#include "helper.hpp"

using namespace Catlass;
using namespace tla;

using Options = SyrkOptions;

template <class Kernel>
CATLASS_GLOBAL __mix__(1, 2) void SyrkKernel(typename Kernel::Params params)
{
    Kernel kernel;
    kernel(params);
}

static void Run(const Options& options)
{
    aclrtStream stream{nullptr};

    ACL_CHECK(aclInit(nullptr));
    ACL_CHECK(aclrtSetDevice(options.deviceId));
    ACL_CHECK(aclrtCreateStream(&stream));

    const auto releaseAclEarly = [&]() {
        ACL_CHECK(aclrtDestroyStream(stream));
        ACL_CHECK(aclrtResetDevice(options.deviceId));
        ACL_CHECK(aclFinalize());
    };

    // D = alpha * X * X^T + beta * Y, X: [batch, M, K], Y/D: [batch, M, M]
    uint32_t batch = options.batchCount;
    uint32_t m = options.problemShape.m();
    uint32_t k = options.problemShape.k();

    using ElementX = bfloat16_t;
    using ElementY = bfloat16_t;
    using ElementD = bfloat16_t;
    using ElementP = float; // workspace tile dtype

    // Host-side tags for golden only; device layouts are fixed inside the kernel.
    using LayoutTagX = layout::RowMajor;
    using LayoutTagXt = layout::ColumnMajor;
    using LayoutTagY = layout::RowMajor;

    LayoutTagX tagX = LayoutTagX::MakeLayout<ElementX>(m, k);
    LayoutTagXt tagXt = LayoutTagXt::MakeLayout<ElementX>(k, m);
    LayoutTagY tagY = LayoutTagY::MakeLayout<ElementY>(m, m);

    size_t lenX = static_cast<size_t>(batch) * tagX.Capacity();
    size_t lenY = static_cast<size_t>(batch) * tagY.Capacity();

    size_t sizeX = lenX * sizeof(ElementX);
    size_t sizeY = lenY * sizeof(ElementY);

    std::vector<bfloat16> hostX(lenX);
    golden::FillRandomData<bfloat16>(hostX, -5.0f, 5.0f);
    std::vector<bfloat16> hostY(lenY);
    golden::FillRandomData<bfloat16>(hostY, -5.0f, 5.0f);

    uint8_t* deviceX{nullptr};
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void**>(&deviceX), sizeX, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemcpy(deviceX, sizeX, hostX.data(), sizeX, ACL_MEMCPY_HOST_TO_DEVICE));

    uint8_t* deviceY{nullptr};
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void**>(&deviceY), sizeY, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemcpy(deviceY, sizeY, hostY.data(), sizeY, ACL_MEMCPY_HOST_TO_DEVICE));

    uint8_t* deviceD{nullptr};
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void**>(&deviceD), sizeY, ACL_MEM_MALLOC_HUGE_FIRST));

    uint8_t* deviceWorkspace{nullptr};

    auto aicCoreNum = platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreNumAic();

    using L1TileShape = Shape<Int<256>, Int<256>, Int<128>>;
    using L0TileShape = Shape<Int<256>, Int<256>, Int<64>>;

    // BlockMmad dual-writes float P / P^T into GM workspace; AIV applies AXPBY.
    using BlockMmad = Gemm::Block::BlockMmadSyrkTla<L1TileShape, L0TileShape, ElementX, ElementP>;
    using BlockEpilogue = Epilogue::Block::BlockEpilogueSyrkAxpby<Arch::Ascend950, ElementY, ElementD>;

    using BlockScheduler = Gemm::Block::SyrkBlockScheduler;
    using MatmulKernel = Gemm::Kernel::BasicSyrkWorkspaceTla<BlockMmad, BlockEpilogue, BlockScheduler>;

    BlockScheduler scheduler(GemmCoord{m, m, k}, Catlass::MakeCoord(MatmulKernel::L1_TILE_M, MatmulKernel::L1_TILE_N));
    uint32_t taskNum = batch * scheduler.GetCoreLoops();
    uint32_t aicCoreUsed = min(aicCoreNum, taskNum);

    typename MatmulKernel::Arguments arguments{batch,   GemmCoord{m, m, k}, deviceX,      deviceY,
                                               deviceD, options.alpha,      options.beta, aicCoreUsed};
    if (!MatmulKernel::CanImplement(arguments)) {
        std::cerr << "MatmulKernel can not implement the arguments: m must equal n." << std::endl;
        releaseAclEarly();
        return;
    }

    size_t sizeWorkspace = MatmulKernel::GetWorkspaceSize(arguments);
    if (sizeWorkspace > 0) {
        ACL_CHECK(aclrtMalloc(reinterpret_cast<void**>(&deviceWorkspace), sizeWorkspace, ACL_MEM_MALLOC_HUGE_FIRST));
    }
    auto params = MatmulKernel::ToUnderlyingArguments(arguments, deviceWorkspace);
    SyrkKernel<MatmulKernel><<<aicCoreUsed, nullptr, stream>>>(params);
    ACL_CHECK(aclrtSynchronizeStream(stream));

    std::vector<bfloat16> hostD(lenY);
    ACL_CHECK(aclrtMemcpy(hostD.data(), sizeY, deviceD, sizeY, ACL_MEMCPY_DEVICE_TO_HOST));

    std::vector<float> hostGolden(lenY);
    std::vector<GemmCoord> problemShapeList(batch, GemmCoord{m, m, k});
    std::vector<float> alphaList(batch, options.alpha);
    std::vector<float> betaList(batch, options.beta);
    std::vector<LayoutTagX> tagXList(batch, tagX);
    std::vector<LayoutTagXt> tagXtList(batch, tagXt);
    std::vector<LayoutTagY> tagYList(batch, tagY);
    golden::ComputeGroupGemm(
        batch, problemShapeList, alphaList, betaList, hostX, tagXList, hostX, tagXtList, hostY, tagYList, hostGolden,
        tagYList);

    std::vector<uint64_t> errorIndices = golden::CompareData(hostD, hostGolden, k);
    if (errorIndices.empty()) {
        std::cout << "Compare success." << std::endl;
    } else {
        std::cerr << "Compare failed. Error count: " << errorIndices.size() << std::endl;
    }

    ACL_CHECK(aclrtFree(deviceX));
    ACL_CHECK(aclrtFree(deviceY));
    ACL_CHECK(aclrtFree(deviceD));
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
    Run(options);
    return 0;
}
