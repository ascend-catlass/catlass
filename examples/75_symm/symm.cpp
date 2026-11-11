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
#include "catlass/gemm/kernel/symm_tla.hpp"

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

#include <acl/acl_rt.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <string>
#include <unistd.h>

using namespace Catlass;
using namespace tla;

using Options = GemmOptions;

struct SymmOptions : Options {
    int32_t symmSide{0};
    int32_t symmFill{1};
};

enum class SymmSide : int32_t
{
    LEFT = 0,
    RIGHT = 1
};
enum class SymmFillMode : int32_t
{
    LOWER = 0,
    UPPER = 1
};

static inline aclError AclCheckWithMsg(aclError error, const char* expr, const char* file, int line)
{
    if (error != ACL_ERROR_NONE) {
        const char* msg = aclGetRecentErrMsg();
        std::cerr << file << ":" << line << " " << expr << " aclError:" << error;
        if (msg != nullptr)
            std::cerr << " (" << msg << ")";
        std::cerr << std::endl;
    }
    return error;
}
#define ACL_CHECK_MSG(expr) AclCheckWithMsg((expr), #expr, __FILE__, __LINE__)

static void Run(const SymmOptions& opt)
{
    uint32_t m = opt.problemShape.m(), n = opt.problemShape.n(), k = opt.problemShape.k();
    SymmSide symmSide = (opt.symmSide == 0) ? SymmSide::LEFT : SymmSide::RIGHT;
    SymmFillMode symmFillMode = (opt.symmFill != 0) ? SymmFillMode::UPPER : SymmFillMode::LOWER;

    if (symmSide == SymmSide::LEFT && m != k) {
        std::cerr << "Left symm: M == K required." << std::endl;
        return;
    }
    if (symmSide == SymmSide::RIGHT && k != n) {
        std::cerr << "Right symm: K == N required." << std::endl;
        return;
    }

    aclrtStream stream{nullptr};
    ACL_CHECK(aclInit(nullptr));
    ACL_CHECK(aclrtSetDevice(opt.deviceId));
    ACL_CHECK(aclrtCreateStream(&stream));

    using ElementA = float;
    using ElementB = float;
    using ElementC = float;
    size_t lenA = static_cast<size_t>(m) * k, lenB = static_cast<size_t>(k) * n, lenC = static_cast<size_t>(m) * n;
    size_t sizeA = lenA * sizeof(ElementA), sizeB = lenB * sizeof(ElementB), sizeC = lenC * sizeof(ElementC);

    using LayoutA = layout::RowMajor;
    using LayoutB = layout::RowMajor;
    using LayoutC = layout::RowMajor;
    LayoutA layoutA = LayoutA::template MakeLayout<ElementA>(m, k);
    LayoutB layoutB = LayoutB::template MakeLayout<ElementB>(k, n);
    LayoutC layoutC{m, n};

    std::vector<float> hostA(lenA), hostB(lenB);
    bool upperStorage = (symmFillMode == SymmFillMode::UPPER);
    if (symmSide == SymmSide::LEFT) {
        golden::FillSymmRandomData<float>(hostA, m, upperStorage, -5.0f, 5.0f);
        golden::FillRandomData<float>(hostB, -5.0f, 5.0f);
    } else {
        golden::FillRandomData<float>(hostA, -5.0f, 5.0f);
        golden::FillSymmRandomData<float>(hostB, k, upperStorage, -5.0f, 5.0f);
    }

    uint8_t *deviceA{nullptr}, *deviceB{nullptr}, *deviceC{nullptr};
    ACL_CHECK_MSG(aclrtMalloc(reinterpret_cast<void**>(&deviceA), sizeA, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK_MSG(aclrtMemcpy(deviceA, sizeA, hostA.data(), sizeA, ACL_MEMCPY_HOST_TO_DEVICE));
    ACL_CHECK_MSG(aclrtMalloc(reinterpret_cast<void**>(&deviceB), sizeB, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK_MSG(aclrtMemcpy(deviceB, sizeB, hostB.data(), sizeB, ACL_MEMCPY_HOST_TO_DEVICE));
    ACL_CHECK_MSG(aclrtMalloc(reinterpret_cast<void**>(&deviceC), sizeC, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK_MSG(aclrtMemset(deviceC, sizeC, 0, sizeC));

    auto aicCoreNum = platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreNumAic();
    using ArchTag = Arch::AtlasA2;

    if (symmSide == SymmSide::LEFT) {
        using L1S = Shape<Int<128>, Int<256>, Int<128>>;
        using L0S = Shape<Int<128>, Int<256>, Int<32>>;
        auto lAP = tla::MakeLayout<ElementA, layout::RowMajor>(m, k);
        auto lAQ = tla::MakeLayout<ElementA, layout::ColumnMajor>(m, k);
        auto lB = tla::MakeLayout<ElementB, layout::RowMajor>(k, n);
        auto lC = tla::MakeLayout<ElementC, layout::RowMajor>(m, n);
        // ENABLE_UNIT_FLAG: enables unit flag for L0C→GM copy and MMA ops
        using DP = Gemm::MmadPingpongSymmLeft<ArchTag, true>;
        using TCP = Gemm::Tile::PackedTileCopyTla<
            ArchTag, ElementA, layout::RowMajor, ElementB, layout::RowMajor, ElementC, layout::RowMajor>;
        using TCQ = Gemm::Tile::PackedTileCopyTla<
            ArchTag, ElementA, layout::ColumnMajor, ElementB, layout::RowMajor, ElementC, layout::RowMajor>;
        using Block =
            Gemm::Block::BlockMmadPingpongSymmLeftTla<DP, L1S, L0S, ElementA, ElementB, ElementC, void, TCP, TCQ>;

        using Sched = typename Gemm::Block::GemmIdentityBlockSwizzle<3, 0>;
        if (symmFillMode == SymmFillMode::UPPER) {
            using Kern = Gemm::Kernel::SymmMatmulTlaSingleKernelProducer<
                Gemm::Kernel::SymmMatmulSide::LEFT, Gemm::Kernel::SymmMatmulFillMode::UPPER, Block, Sched>;
            typename Kern::Arguments args{opt.problemShape, deviceA, lAP, lAQ, deviceB, lB, deviceC, lC};
            Gemm::Device::DeviceGemm<Kern> op;
            op.Initialize(args, nullptr);
            op(stream, aicCoreNum);
        } else {
            using Kern = Gemm::Kernel::SymmMatmulTlaSingleKernelProducer<
                Gemm::Kernel::SymmMatmulSide::LEFT, Gemm::Kernel::SymmMatmulFillMode::LOWER, Block, Sched>;
            typename Kern::Arguments args{opt.problemShape, deviceA, lAP, lAQ, deviceB, lB, deviceC, lC};
            Gemm::Device::DeviceGemm<Kern> op;
            op.Initialize(args, nullptr);
            op(stream, aicCoreNum);
        }
        ACL_CHECK_MSG(aclrtSynchronizeStream(stream));
    } else {
        using L1S = Shape<Int<256>, Int<128>, Int<128>>;
        using L0S = Shape<Int<256>, Int<128>, Int<32>>;
        auto lA = tla::MakeLayout<ElementA, layout::RowMajor>(m, k);
        auto lBP = tla::MakeLayout<ElementB, layout::RowMajor>(k, n);
        auto lBQ = tla::MakeLayout<ElementB, layout::ColumnMajor>(k, n);
        auto lC = tla::MakeLayout<ElementC, layout::RowMajor>(m, n);
        // ENABLE_UNIT_FLAG: enables unit flag for L0C→GM copy and MMA ops
        using DP = Gemm::MmadPingpongSymmRight<ArchTag, true>;
        using TCP = Gemm::Tile::PackedTileCopyTla<
            ArchTag, ElementA, layout::RowMajor, ElementB, layout::RowMajor, ElementC, layout::RowMajor>;
        using TCQ = Gemm::Tile::PackedTileCopyTla<
            ArchTag, ElementA, layout::RowMajor, ElementB, layout::ColumnMajor, ElementC, layout::RowMajor>;
        using Block =
            Gemm::Block::BlockMmadPingpongSymmRightTla<DP, L1S, L0S, ElementA, ElementB, ElementC, void, TCP, TCQ>;

        using Sched = typename Gemm::Block::GemmIdentityBlockSwizzle<3, 0>;
        if (symmFillMode == SymmFillMode::UPPER) {
            using Kern = Gemm::Kernel::SymmMatmulTlaSingleKernelProducer<
                Gemm::Kernel::SymmMatmulSide::RIGHT, Gemm::Kernel::SymmMatmulFillMode::UPPER, Block, Sched>;
            typename Kern::Arguments args{opt.problemShape, deviceA, lBP, lBQ, deviceB, lA, deviceC, lC};
            Gemm::Device::DeviceGemm<Kern> op;
            op.Initialize(args, nullptr);
            op(stream, aicCoreNum);
        } else {
            using Kern = Gemm::Kernel::SymmMatmulTlaSingleKernelProducer<
                Gemm::Kernel::SymmMatmulSide::RIGHT, Gemm::Kernel::SymmMatmulFillMode::LOWER, Block, Sched>;
            typename Kern::Arguments args{opt.problemShape, deviceA, lBP, lBQ, deviceB, lA, deviceC, lC};
            Gemm::Device::DeviceGemm<Kern> op;
            op.Initialize(args, nullptr);
            op(stream, aicCoreNum);
        }
        ACL_CHECK_MSG(aclrtSynchronizeStream(stream));
    }

    std::vector<ElementC> hostC(lenC);
    ACL_CHECK(aclrtMemcpy(hostC.data(), sizeC, deviceC, sizeC, ACL_MEMCPY_DEVICE_TO_HOST));

    std::vector<float> hostGolden(lenC);
    golden::ComputeMatmul(opt.problemShape, hostA, layoutA, hostB, layoutB, hostGolden, layoutC);
    std::vector<uint64_t> errorIndices = golden::CompareData(hostC, hostGolden, k);
    if (errorIndices.empty())
        std::cout << "Compare success." << std::endl;
    else
        std::cerr << "Compare failed. Error count: " << errorIndices.size() << std::endl;

    ACL_CHECK(aclrtFree(deviceA));
    ACL_CHECK(aclrtFree(deviceB));
    ACL_CHECK(aclrtFree(deviceC));
    ACL_CHECK(aclrtDestroyStream(stream));
    ACL_CHECK(aclrtResetDevice(opt.deviceId));
    ACL_CHECK(aclFinalize());
}

int main(int argc, const char** argv)
{
    if (argc < 4) {
        std::cerr << "Usage: " << argv[0] << " <m> <n> <k> [device_id] [symm_side: 0=left, 1=right] "
                  << "[symm_fill: 0=lower, 1=upper]" << std::endl;
        return 1;
    }

    SymmOptions opt;
    opt.problemShape.m() = std::atoi(argv[1]);
    opt.problemShape.n() = std::atoi(argv[2]);
    opt.problemShape.k() = std::atoi(argv[3]);
    opt.deviceId = (argc >= 5) ? std::atoi(argv[4]) : 0;
    opt.symmSide = (argc >= 6) ? std::atoi(argv[5]) : 0;
    opt.symmFill = (argc >= 7) ? std::atoi(argv[6]) : 1;
    Run(opt);
    return 0;
}
